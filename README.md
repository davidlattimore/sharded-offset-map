# Sharded Offset Map

This crate provides what is effectively a `HashMap<u64, T>`. To be effective, the keys need to have
a relatively uniform distribution. The user of the map selects a block size via a const generic. The
map can then store up to 12 key-value pairs within each block. So for example, if the block size is
256, then the key space is divided into blocks of 256. If any block needs more than 12 key-value
pairs, it's up to the user of the map to store those elsewhere.

Writes to the map must be done in order, however the map can be split into arbitrary sized shards
with each shard being written concurrently from a separate thread.

```rust
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
```
