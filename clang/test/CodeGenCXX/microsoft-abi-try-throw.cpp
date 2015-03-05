// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i386-pc-win32 -mconstructor-aliases -fcxx-exceptions -fexceptions -fno-rtti -DTRY   | FileCheck %s -check-prefix=TRY
// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i386-pc-win32 -mconstructor-aliases -fcxx-exceptions -fexceptions -fno-rtti -DTHROW | FileCheck %s -check-prefix=THROW

// THROW-DAG: @"\01??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { i8** @"\01??_7type_info@@6B@", i8* null, [3 x i8] c".H\00" }, comdat
// THROW-DAG: @"_CT??_R0H@84" = linkonce_odr unnamed_addr constant %eh.CatchableType { i32 1, i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*), i32 0, i32 -1, i32 0, i32 4, i8* null }, comdat
// THROW-DAG: @_CTA1H = linkonce_odr unnamed_addr constant %eh.CatchableTypeArray.1 { i32 1, [1 x %eh.CatchableType*] [%eh.CatchableType* @"_CT??_R0H@84"] }, comdat
// THROW-DAG: @_TI1H = linkonce_odr unnamed_addr constant %eh.ThrowInfo { i32 0, i8* null, i8* null, i8* bitcast (%eh.CatchableTypeArray.1* @_CTA1H to i8*) }, comdat

void external();

inline void not_emitted() {
  throw int(13); // no error
}

int main() {
  int rv = 0;
#ifdef TRY
  try {
    external(); // TRY: invoke void @"\01?external@@YAXXZ"
  } catch (int) {
    rv = 1;
    // TRY: call void @llvm.eh.begincatch(i8* %{{.*}}, i8* %{{.*}})
    // TRY: call void @llvm.eh.endcatch()
  }
#endif
#ifdef THROW
  // THROW: store i32 42, i32* %[[mem_for_throw:.*]]
  // THROW: %[[cast:.*]] = bitcast i32* %[[mem_for_throw]] to i8*
  // THROW: call void @_CxxThrowException(i8* %[[cast]], %eh.ThrowInfo* @_TI1H)
  throw int(42);
#endif
  return rv;
}
