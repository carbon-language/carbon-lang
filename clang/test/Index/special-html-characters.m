// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng -triple x86_64-apple-darwin10 %s > %t/out
// RUN: FileCheck %s < %t/out
// rdar://13067629

// Ensure that XML we generate is not invalid.
// RUN: FileCheck %s -check-prefix=WRONG < %t/out
// WRONG-NOT: CommentXMLInvalid

// XFAIL: valgrind

// rdar://12392215
@interface I
@end

@implementation I
/*!
	&copy; the copyright symbol
	&trade; the trade mark symbol
        &reg; the registered trade mark symbol
	&nbsp; a non breakable space.
        &Delta; Greek letter Delta Δ.
        &Gamma; Greek letter Gamma Γ.
 */
- (void)phoneHome:(id)sender {

}
@end
// CHECK: FullCommentAsHTML=[<p class="para-brief">\t© the copyright symbol\t™ the trade mark symbol        ® the registered trade mark symbol\t  a non breakable space.        Δ Greek letter Delta Δ.        Γ Greek letter Gamma Γ. </p>] FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}special-html-characters.m" line="[[@LINE-4]]" column="1"><Name>phoneHome:</Name><USR>c:objc(cs)I(im)phoneHome:</USR><Declaration>- (void)phoneHome:(id)sender;</Declaration><Abstract><Para>\t© the copyright symbol\t™ the trade mark symbol        ® the registered trade mark symbol\t  a non breakable space.        Δ Greek letter Delta Δ.        Γ Greek letter Gamma Γ. </Para></Abstract></Function>] 
