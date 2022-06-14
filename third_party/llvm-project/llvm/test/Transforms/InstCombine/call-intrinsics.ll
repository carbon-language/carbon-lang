; RUN: opt < %s -passes=instcombine | llvm-dis

@X = global i8 0
@Y = global i8 12

declare void @llvm.memmove.p0i8.p0i8.i32(i8*, i8*, i32, i1)

declare void @llvm.memcpy.p0i8.p0i8.i32(i8*, i8*, i32, i1)

declare void @llvm.memset.p0i8.i32(i8*, i8, i32, i1)

define void @zero_byte_test() {
  ; These process zero bytes, so they are a noop.
  call void @llvm.memmove.p0i8.p0i8.i32(i8* @X, i8* @Y, i32 0, i1 false )
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* @X, i8* @Y, i32 0, i1 false )
  call void @llvm.memset.p0i8.i32(i8* @X, i8 123, i32 0, i1 false )
  ret void
}

