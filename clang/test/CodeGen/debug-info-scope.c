// RUN: %clang_cc1 -g -emit-llvm < %s | FileCheck %s

// Make sure that the debug info of the local variable d does not shadow
// the global variable d
// CHECK: [ DW_TAG_variable ] [d] [line 6] [def]
const int d = 100;

// Two variables with same name in separate scope.
// Radar 8330217.
int main() {
// CHECK-NOT: [ DW_TAG_variable ] [d] [line 13]
// CHECK: [ DW_TAG_auto_variable ] [d] [line 13]
	const int d = 4;
	int j = 0;
	int k = 0;
// CHECK: DW_TAG_auto_variable ] [i]
// CHECK-NEXT: DW_TAG_lexical_block
	for (int i = 0; i < 10; i++)
		j++;
// CHECK: DW_TAG_auto_variable ] [i]
// CHECK-NEXT: DW_TAG_lexical_block
	for (int i = 0; i < 10; i++)
		k++;
	return d; // make a reference to d so that its debug info gets included
}
