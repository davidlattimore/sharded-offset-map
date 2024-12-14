// An `OffsetMap` is effectively a `HashMap<u64,T>` but with different performance characteristics.
// The input space is divided into blocks of size `BLOCK_SIZE`. Each block can map up to 12 input
// values to output values. If more are needed, it's up to the user to store them elsewhere, e.g. in
// an actual hashmap. `T` should ideally be the size of a `u32`, otherwise a block won't fit neatly
// in a cache line and performance might be degraded.
#[derive(Default)]
pub struct OffsetMap<T, const BLOCK_SIZE: u64> {
    blocks: Vec<Block<T, BLOCK_SIZE>>,
}

/// The maximum number of offsets that can be stored in one block. Attempts to add more offsets that
/// this will fail and it's up to the user of the library to store those offsets elsewhere, e.g. in
/// a hashmap.
pub const MAX_KEYS_PER_BLOCK: u8 = 12;

/// A block that maps up to 12 input offsets within a range of S input offsets to T. The size and
/// alignment of this struct is designed to fit into a typical cache line.
#[repr(align(64))]
#[derive(Default)]
pub struct Block<T, const BLOCK_SIZE: u64> {
    /// The low 12 bits indicate which offsets are used.
    used: u16,

    /// The offsets of up to 12 inputs relative to the base of the block.
    relative_offsets: [u8; 12],

    /// The output values corresponding to our relative offsets.
    values: [T; 12],
}

/// The offset of a particular value within a shard. Can be used to update a value some time after
/// an insertion.
#[derive(Clone, Copy)]
pub struct OffsetInShard(u32);
impl OffsetInShard {
    fn new(block_index: usize, index_in_block: u8) -> Self {
        Self(((block_index as u32) << 4) | index_in_block as u32)
    }

    fn block_index(self) -> usize {
        (self.0 >> 4) as usize
    }

    fn index_in_block(self) -> usize {
        (self.0 & 0xf) as usize
    }
}

pub struct ShardedWriter<'map, T, const BLOCK_SIZE: u64> {
    vec_writer: sharded_vec_writer::VecWriter<'map, Block<T, BLOCK_SIZE>>,
    base: u64,
}

pub struct Shard<'map, T, const BLOCK_SIZE: u64> {
    last_key: Option<u64>,
    base: u64,
    key_offset: u64,
    vec_shard: sharded_vec_writer::Shard<'map, Block<T, BLOCK_SIZE>>,
    index_in_block: Option<u8>,
}

/// An error that indicates that an insert failed due to the block for the key being full. The
/// caller should generally store the offset somewhere else, e.g. in a hashmap.
#[derive(Debug)]
pub struct BlockFull;

impl<T: Copy, const BLOCK_SIZE: u64> OffsetMap<T, BLOCK_SIZE> {
    pub fn get(&self, key: u64) -> Option<T> {
        let block_index = key / BLOCK_SIZE;
        let offset_in_block = (key % BLOCK_SIZE) as u8;
        let block = self.blocks.get(block_index as usize)?;
        for (i, offset) in block.relative_offsets.iter().enumerate() {
            if *offset == offset_in_block && block.used & (1 << i) != 0 {
                return Some(block.values[i]);
            }
        }
        None
    }

    /// Start a write. Keys in the final map will range from 0 to `keyspace_size - 1`.
    pub fn start_sharded_write(&mut self, keyspace_size: u64) -> ShardedWriter<T, BLOCK_SIZE> {
        assert_eq!(keyspace_size % BLOCK_SIZE, 0);
        // For now, performing multiple writes is not supported.
        assert_eq!(self.blocks.capacity(), 0);
        let num_blocks = keyspace_size / BLOCK_SIZE;
        self.blocks.reserve(num_blocks as usize);
        ShardedWriter {
            vec_writer: sharded_vec_writer::VecWriter::new(&mut self.blocks),
            base: 0,
        }
    }

    /// Iterate through all key-value pairs in the map.
    pub fn iter(&self) -> impl Iterator<Item = (u64, T)> + '_ {
        self.blocks
            .iter()
            .enumerate()
            .flat_map(|(block_num, block)| {
                (0..block.used.count_ones()).map(move |i| {
                    (
                        block_num as u64 * BLOCK_SIZE + i as u64,
                        block.values[i as usize],
                    )
                })
            })
    }
}

impl<'map, T: Default, const BLOCK_SIZE: u64> ShardedWriter<'map, T, BLOCK_SIZE> {
    /// Take a shard into which writing can occur. `size` must be an exact multiple of `BLOCK_SIZE`.
    /// Taken shards should be returned in the order they were taken by calling `return_shard`.
    pub fn take_shard(&mut self, size: u64) -> Shard<'map, T, BLOCK_SIZE> {
        assert_eq!(size % BLOCK_SIZE, 0);
        let num_blocks = size / BLOCK_SIZE;
        let vec_shard = self.vec_writer.take_shard(num_blocks as usize);
        let key_offset = self.base;
        self.base += size;
        Shard {
            base: key_offset,
            key_offset,
            vec_shard,
            index_in_block: None,
            last_key: key_offset.checked_sub(1),
        }
    }

    pub fn return_shard(&mut self, mut shard: Shard<'map, T, BLOCK_SIZE>) {
        shard.finish();
        self.vec_writer.return_shard(shard.vec_shard);
    }
}

impl<T: Default, const BLOCK_SIZE: u64> Shard<'_, T, BLOCK_SIZE> {
    /// Inserts the supplied key-value pair. If the block is already full, then returns an error.
    /// Panics if the supplied key falls outside of the range of the current shard or if its less
    /// than or equal to the last key supplied.
    pub fn insert(&mut self, key: u64, value: T) -> Result<OffsetInShard, BlockFull> {
        if self.last_key.is_some_and(|last_key| last_key >= key) {
            if key < self.key_offset {
                panic!(
                    "Key {key} supplied when block starts at {}",
                    self.key_offset
                );
            }
            panic!(
                "Keys must always advance. Got {} after {}",
                key,
                self.last_key.unwrap()
            );
        }
        self.last_key = Some(key);

        let mut offset_in_block = key - self.key_offset;

        // See if the key is past the end of the current block.
        while offset_in_block >= BLOCK_SIZE {
            if self.index_in_block.take().is_none() {
                self.vec_shard.push(Block::default());
            }
            offset_in_block -= BLOCK_SIZE;
            self.key_offset += BLOCK_SIZE;
        }

        // Compute which index in our block we'll be writing to.
        let index_in_block = self.index_in_block.unwrap_or_else(|| {
            if self.vec_shard.try_push(Block::default()).is_err() {
                panic!(
                    "Tried to insert key {key} beyond end of shard {}",
                    self.base + self.vec_shard.len() as u64 * BLOCK_SIZE
                );
            }
            0
        });

        if index_in_block == MAX_KEYS_PER_BLOCK {
            return Err(BlockFull);
        }

        let i = usize::from(index_in_block);

        let init_blocks = self.vec_shard.init_mut();
        let block = init_blocks.last_mut().unwrap();

        block.used |= 1 << index_in_block;
        block.relative_offsets[i] = offset_in_block as u8;
        block.values[i] = value;

        let offset = OffsetInShard::new(init_blocks.len() - 1, index_in_block);

        self.index_in_block = Some(index_in_block + 1);

        Ok(offset)
    }

    pub fn get_mut(&mut self, offset: OffsetInShard) -> &mut T {
        &mut self.vec_shard.init_mut()[offset.block_index()].values[offset.index_in_block()]
    }

    /// Returns an iterator over values that have already been written to this shard.
    pub fn iter_mut(&mut self) -> ShardIterMut<T, BLOCK_SIZE> {
        let mut iter = ShardIterMut {
            base: self.base,
            blocks: self.vec_shard.init_mut(),
            used: 0,
            relative_offsets: &[],
            values: &mut [],
        };
        iter.consume_block();
        iter
    }

    /// Consume the remainder of the space allocated to this writer. This needs to be called prior
    /// returning the shard to the `VecWriter`.
    pub fn finish(&mut self) {
        while self.vec_shard.try_push(Block::default()).is_ok() {}
        self.key_offset = u64::MAX;
    }
}

pub struct ShardIterMut<'map, T, const BLOCK_SIZE: u64> {
    base: u64,
    blocks: &'map mut [Block<T, BLOCK_SIZE>],

    used: u16,
    relative_offsets: &'map [u8],
    values: &'map mut [T],
}

impl<T, const BLOCK_SIZE: u64> ShardIterMut<'_, T, BLOCK_SIZE> {
    fn consume_block(&mut self) {
        let blocks = core::mem::take(&mut self.blocks);
        let Some((block, rest)) = blocks.split_first_mut() else {
            return;
        };
        self.blocks = rest;

        self.used = block.used;
        self.relative_offsets = block.relative_offsets.as_slice();
        self.values = block.values.as_mut_slice();
    }
}

impl<'map, T, const BLOCK_SIZE: u64> Iterator for ShardIterMut<'map, T, BLOCK_SIZE> {
    type Item = (u64, &'map mut T);

    fn next(&mut self) -> Option<Self::Item> {
        while self.used == 0 {
            if self.blocks.is_empty() {
                return None;
            }
            self.consume_block();
            self.base += BLOCK_SIZE;
        }

        let values = core::mem::take(&mut self.values);
        let (value, rest) = values.split_first_mut().unwrap();
        self.values = rest;

        let key = self.base + u64::from(self.relative_offsets[0]);
        self.relative_offsets = &self.relative_offsets[1..];

        self.used >>= 1;

        Some((key, value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[track_caller]
    fn check(present: &[(u64, u32)], absent: &[u64]) {
        check_full(present, absent, false);
        check_full(present, absent, true);
    }

    #[track_caller]
    fn check_full(present: &[(u64, u32)], absent: &[u64], multi_shards: bool) {
        let mut map = OffsetMap::<u32, 128>::default();
        let mut writer = map.start_sharded_write(1024);
        let shard_size = if multi_shards { 256 } else { 1024 };
        let mut shard = writer.take_shard(shard_size);

        let mut inserted = Vec::new();

        for (key, value) in present {
            while *key >= shard.base + shard_size {
                let init: Vec<_> = shard.iter_mut().map(|(key, value)| (key, *value)).collect();
                assert_eq!(inserted, init);
                inserted.clear();

                writer.return_shard(shard);
                shard = writer.take_shard(shard_size);
            }
            shard.insert(*key, *value).unwrap();
            inserted.push((*key, *value));
        }

        if !multi_shards {
            let init: Vec<_> = shard.iter_mut().map(|(key, value)| (key, *value)).collect();
            assert_eq!(present, init);
        }

        writer.return_shard(shard);

        for (key, value) in present {
            let actual = map.get(*key);
            assert_eq!(
                actual,
                Some(*value),
                "Value at {key} is {actual:?}, expected {value}"
            );
        }

        for key in absent {
            assert_eq!(map.get(*key), None);
        }
    }

    #[test]
    fn basic_usage() {
        check(&[], &[0, 10, 63, 64, 65]);
        check(&[(0, 0)], &[1]);
        check(&[(0, 1)], &[1]);
        check(&[(1, 10)], &[0, 2]);
        check(
            &[
                (1, 11),
                (3, 13),
                (63, 73),
                (64, 74),
                (65, 75),
                (650, 1234),
                (1000, 1230),
            ],
            &[0, 2, 4, 62, 66],
        );
        check(&[(64, 100)], &[0, 63, 65]);
        check(&[(170, 100)], &[0, 63, 65]);
        check(&[(170, 100)], &[0, 63, 65]);
        check(&[(600, 100)], &[700]);
    }

    #[test]
    fn readme_example() {
        let mut map = OffsetMap::<u32, 128>::default();

        // Start a write, advising the final size of the keyspace of the map.
        let mut writer = map.start_sharded_write(1024);

        // Shards can be whatever size we like, however they must be a multiple of the block size.
        let mut shards = (0..4).map(|_| writer.take_shard(256)).collect::<Vec<_>>();

        // We can use scoped threads to concurrently write to the different map shards.
        std::thread::scope(|s| {
            for (i, shard) in shards.iter_mut().enumerate() {
                s.spawn(move || {
                    for j in 0..24 {
                        if shard.insert((i * 256 + j * 11) as u64, j as u32).is_err() {
                            unimplemented!("out-of-band storage not implemented in this test");
                        }
                    }
                });
            }
        });

        // Shards must be returned to the writer in order.
        for shard in shards {
            writer.return_shard(shard);
        }

        // Once all shards have been returned, we can access our map.
        assert_eq!(map.get(0), Some(0));
        assert_eq!(map.get(1), None);
        assert_eq!(map.get(11), Some(1));
        assert_eq!(map.get(256), Some(0));
        assert_eq!(map.get(278), Some(2));
    }
}
