// RUN: %clang_cc1 -g -emit-llvm < %s | FileCheck %s
// RUN: %clang_cc1 -gline-tables-only -emit-llvm < %s | FileCheck --check-prefix=GMLT %s
// Two variables with same name in separate scope.
// Radar 8330217.
int main() {
	int j = 0;
	int k = 0;
// CHECK: !MDLocalVariable(tag: DW_TAG_auto_variable, name: "i"
// CHECK-NEXT: !MDLexicalBlock(

// FIXME: Looks like we don't actually need both these lexical blocks (disc 2
// just refers to disc 1, nothing actually uses disc 2).
// GMLT-NOT: !MDLexicalBlock
// GMLT: !MDLexicalBlockFile({{.*}}, discriminator: 2)
// GMLT-NOT: !MDLexicalBlock
// GMLT: !MDLexicalBlockFile({{.*}}, discriminator: 1)
// Make sure we don't have any more lexical blocks because we don't need them in
// -gmlt.
// GMLT-NOT: !MDLexicalBlock
	for (int i = 0; i < 10; i++)
		j++;
// CHECK: !MDLocalVariable(tag: DW_TAG_auto_variable, name: "i"
// CHECK-NEXT: !MDLexicalBlock(
// GMLT-NOT: !MDLexicalBlock
	for (int i = 0; i < 10; i++)
		k++;
	return 0;
}
