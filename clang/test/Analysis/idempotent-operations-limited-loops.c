void always_warning() { int *p = 0; *p = 0xDEADBEEF; }

// FIXME: False positive due to loop unrolling.  This should be fixed.

int pr8403()
{
        int i;
        for(i=0; i<10; i++)
        {
                int j;
                for(j=0; j+1<i; j++)
                {
                }
        }
        return 0;
}

// RUN: %clang_cc1 -analyze -analyzer-store=region -analyzer-constraints=range -fblocks -analyzer-opt-analyze-nested-blocks -analyzer-check-objc-mem -analyzer-check-idempotent-operations -analyzer-max-loop 3 %s 2>&1 | FileCheck --check-prefix=Loops3 %s
// RUN: %clang_cc1 -analyze -analyzer-store=region -analyzer-constraints=range -fblocks -analyzer-opt-analyze-nested-blocks -analyzer-check-objc-mem -analyzer-check-idempotent-operations -analyzer-max-loop 4 %s 2>&1 | FileCheck --check-prefix=Loops4 %s
// RUN: %clang_cc1 -analyze -analyzer-store=region -analyzer-constraints=range -fblocks -analyzer-opt-analyze-nested-blocks -analyzer-check-objc-mem -analyzer-check-idempotent-operations %s 2>&1 | FileCheck --check-prefix=LoopsDefault %s

// CHECK-Loops3: :1:37: warning: Dereference of null pointer
// CHECK-Loops3: :11:27: warning: The left operand to '+' is always 0
// CHECK-Loops3: 2 warnings generated
// CHECK-Loops4: :1:37: warning: Dereference of null pointer
// CHECK-Loops4: 1 warning generated.
// CHECK-LoopsDefault: :1:37: warning: Dereference of null pointer
// CHECK-LoopsDefault: 1 warning generated.

