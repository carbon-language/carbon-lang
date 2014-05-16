@zed = global i32 42
@foo = alias i32* @zed
@foo2 = alias i16, i32* @zed
