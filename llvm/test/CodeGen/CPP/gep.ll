; RUN: llc -march=cpp -o - %s | FileCheck %s

define void @f1(i32* %addr) {
  %x = getelementptr i32, i32* %addr, i32 1
; CHECK: ConstantInt* [[INT_1:.*]] = ConstantInt::get(mod->getContext(), APInt(32, StringRef("1"), 10));
; CHECK: GetElementPtrInst::Create(IntegerType::get(mod->getContext(), 32), ptr_addr,
; CHECK-NEXT:  [[INT_1]]
; CHECK-NEXT: }, "x", label_3);
  ret void
}
