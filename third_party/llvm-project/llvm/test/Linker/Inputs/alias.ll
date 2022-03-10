@zed = global i32 42
@foo = alias i32, i32* @zed
@foo2 = alias i16, bitcast (i32* @zed to i16*)
