target datalayout = "e-m:w-p:32:32-i64:64-f80:32-n8:16:32-S32"

$foo = comdat largest

@zed = external constant i8
@some_name = private unnamed_addr constant [2 x i8*] [i8* @zed, i8* bitcast (void ()* @bar to i8*)], comdat($foo)
@foo = alias i8*, getelementptr([2 x i8*], [2 x i8*]* @some_name, i32 0, i32 1)

declare void @bar() unnamed_addr
