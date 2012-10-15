// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng %s > %t/out
// RUN: FileCheck %s < %t/out
// Test to search overridden methods for documentation when overriding method has none. rdar://12378793

// Ensure that XML we generate is not invalid.
// RUN: FileCheck %s -check-prefix=WRONG < %t/out
// WRONG-NOT: CommentXMLInvalid

@protocol P
- (void)METH:(id)PPP;
@end

@interface Root<P>
/**
 * \param[in] AAA ZZZ
 */
- (void)METH:(id)AAA;
@end

@interface Sub : Root
@end

@interface Sub (CAT)
- (void)METH:(id)BBB;
@end

@implementation Sub(CAT)
- (void)METH:(id)III {}
@end

// CHECK: FullCommentAsHTML=[<dl><dt class="param-name-index-0">AAA</dt><dd class="param-descr-index-0"> ZZZ </dd></dl>] FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}overriding-method-comments.mm" line="19" column="1"><Name>METH:</Name><USR>c:objc(cs)Root(im)METH:</USR><Parameters><Parameter><Name>AAA</Name><Index>0</Index><Direction isExplicit="1">in</Direction><Discussion><Para> ZZZ </Para></Discussion></Parameter></Parameters></Function>

// CHECK: FullCommentAsHTML=[<dl><dt class="param-name-index-0">BBB</dt><dd class="param-descr-index-0"> ZZZ </dd></dl>] FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}overriding-method-comments.mm" line="26" column="1"><Name>METH:</Name><USR>c:objc(cs)Root(im)METH:</USR><Parameters><Parameter><Name>BBB</Name><Index>0</Index><Direction isExplicit="1">in</Direction><Discussion><Para> ZZZ </Para></Discussion></Parameter></Parameters></Function>

// CHECK: FullCommentAsHTML=[<dl><dt class="param-name-index-0">III</dt><dd class="param-descr-index-0"> ZZZ </dd></dl>] FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}overriding-method-comments.mm" line="30" column="1"><Name>METH:</Name><USR>c:objc(cs)Root(im)METH:</USR><Parameters><Parameter><Name>III</Name><Index>0</Index><Direction isExplicit="1">in</Direction><Discussion><Para> ZZZ </Para></Discussion></Parameter></Parameters></Function>

@interface Redec : Root
@end

@interface Redec()
/**
 * \param[in] AAA input value  
 * \param[out] CCC output value is int 
 * \param[in] BBB 2nd input value is double 
 */
- (void)EXT_METH:(id)AAA : (double)BBB : (int)CCC;
@end

@implementation Redec
- (void)EXT_METH:(id)PPP : (double)QQQ : (int)RRR {}
@end

// CHECK: FullCommentAsHTML=[<dl><dt class="param-name-index-0">AAA</dt><dd class="param-descr-index-0"> input value   </dd><dt class="param-name-index-1">BBB</dt><dd class="param-descr-index-1"> 2nd input value is double  </dd><dt class="param-name-index-2">CCC</dt><dd class="param-descr-index-2"> output value is int  </dd></dl>] FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}overriding-method-comments.mm" line="48" column="1"><Name>EXT_METH:::</Name><USR>c:objc(cs)Redec(im)EXT_METH:::</USR><Parameters><Parameter><Name>AAA</Name><Index>0</Index><Direction isExplicit="1">in</Direction><Discussion><Para> input value   </Para></Discussion></Parameter><Parameter><Name>BBB</Name><Index>1</Index><Direction isExplicit="1">in</Direction><Discussion><Para> 2nd input value is double  </Para></Discussion></Parameter><Parameter><Name>CCC</Name><Index>2</Index><Direction isExplicit="1">out</Direction><Discussion><Para> output value is int  </Para></Discussion></Parameter></Parameters></Function>

// CHECK: FullCommentAsHTML=[<dl><dt class="param-name-index-0">PPP</dt><dd class="param-descr-index-0"> input value   </dd><dt class="param-name-index-1">QQQ</dt><dd class="param-descr-index-1"> 2nd input value is double  </dd><dt class="param-name-index-2">RRR</dt><dd class="param-descr-index-2"> output value is int  </dd></dl>] FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}overriding-method-comments.mm" line="52" column="1"><Name>EXT_METH:::</Name><USR>c:objc(cs)Redec(im)EXT_METH:::</USR><Parameters><Parameter><Name>PPP</Name><Index>0</Index><Direction isExplicit="1">in</Direction><Discussion><Para> input value   </Para></Discussion></Parameter><Parameter><Name>QQQ</Name><Index>1</Index><Direction isExplicit="1">in</Direction><Discussion><Para> 2nd input value is double  </Para></Discussion></Parameter><Parameter><Name>RRR</Name><Index>2</Index><Direction isExplicit="1">out</Direction><Discussion><Para> output value is int  </Para></Discussion></Parameter></Parameters></Function>

struct Base {
  /// \brief Does something.
  /// \param AAA argument to foo_pure.
  virtual void foo_pure(int AAA) = 0;

  /// \brief Does something.
  /// \param BBB argument to defined virtual.
  virtual void foo_inline(int BBB) {}

  /// \brief Does something.
  /// \param CCC argument to undefined virtual.
  virtual void foo_outofline(int CCC);
};

void Base::foo_outofline(int RRR) {}

struct Derived : public Base {
  virtual void foo_pure(int PPP);

  virtual void foo_inline(int QQQ) {}
};

/// \brief Does something.
/// \param DDD a value.
void foo(int DDD);

void foo(int SSS) {}

/// \brief Does something.
/// \param EEE argument to function decl. 
void foo1(int EEE);

void foo1(int TTT);

// CHECK: FullCommentAsHTML=[<p class="para-brief"> Does something. </p><dl><dt class="param-name-index-0">AAA</dt><dd class="param-descr-index-0"> argument to foo_pure.</dd></dl>] FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}overriding-method-comments.mm" line="62" column="16"><Name>foo_pure</Name><USR>c:@S@Base@F@foo_pure#I#</USR><Abstract><Para> Does something. </Para></Abstract><Parameters><Parameter><Name>AAA</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> argument to foo_pure.</Para></Discussion></Parameter></Parameters></Function>

// CHECK: FullCommentAsHTML=[<p class="para-brief"> Does something. </p><dl><dt class="param-name-index-0">BBB</dt><dd class="param-descr-index-0"> argument to defined virtual.</dd></dl>] FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}overriding-method-comments.mm" line="66" column="16"><Name>foo_inline</Name><USR>c:@S@Base@F@foo_inline#I#</USR><Abstract><Para> Does something. </Para></Abstract><Parameters><Parameter><Name>BBB</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> argument to defined virtual.</Para></Discussion></Parameter></Parameters></Function>

// CHECK: FullCommentAsHTML=[<p class="para-brief"> Does something. </p><dl><dt class="param-name-index-0">CCC</dt><dd class="param-descr-index-0"> argument to undefined virtual.</dd></dl>] FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}overriding-method-comments.mm" line="70" column="16"><Name>foo_outofline</Name><USR>c:@S@Base@F@foo_outofline#I#</USR><Abstract><Para> Does something. </Para></Abstract><Parameters><Parameter><Name>CCC</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> argument to undefined virtual.</Para></Discussion></Parameter></Parameters></Function>

// CHECK: FullCommentAsHTML=[<p class="para-brief"> Does something. </p><dl><dt class="param-name-index-0">RRR</dt><dd class="param-descr-index-0"> argument to undefined virtual.</dd></dl>] FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}overriding-method-comments.mm" line="73" column="12"><Name>foo_outofline</Name><USR>c:@S@Base@F@foo_outofline#I#</USR><Abstract><Para> Does something. </Para></Abstract><Parameters><Parameter><Name>RRR</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> argument to undefined virtual.</Para></Discussion></Parameter></Parameters></Function>

// CHECK: FullCommentAsHTML=[<p class="para-brief"> Does something. </p><dl><dt class="param-name-index-0">PPP</dt><dd class="param-descr-index-0"> argument to foo_pure.</dd></dl>] FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}overriding-method-comments.mm" line="76" column="16"><Name>foo_pure</Name><USR>c:@S@Base@F@foo_pure#I#</USR><Abstract><Para> Does something. </Para></Abstract><Parameters><Parameter><Name>PPP</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> argument to foo_pure.</Para></Discussion></Parameter></Parameters></Function>

// CHECK: FullCommentAsHTML=[<p class="para-brief"> Does something. </p><dl><dt class="param-name-index-0">QQQ</dt><dd class="param-descr-index-0"> argument to defined virtual.</dd></dl>] FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}overriding-method-comments.mm" line="78" column="16"><Name>foo_inline</Name><USR>c:@S@Base@F@foo_inline#I#</USR><Abstract><Para> Does something. </Para></Abstract><Parameters><Parameter><Name>QQQ</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> argument to defined virtual.</Para></Discussion></Parameter></Parameters></Function>

// CHECK: FullCommentAsHTML=[<p class="para-brief"> Does something. </p><dl><dt class="param-name-index-0">DDD</dt><dd class="param-descr-index-0"> a value.</dd></dl>] FullCommentAsXML=[<Function file="{{[^"]+}}overriding-method-comments.mm" line="83" column="6"><Name>foo</Name><USR>c:@F@foo#I#</USR><Abstract><Para> Does something. </Para></Abstract><Parameters><Parameter><Name>DDD</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> a value.</Para></Discussion></Parameter></Parameters></Function>

// CHECK: FullCommentAsHTML=[<p class="para-brief"> Does something. </p><dl><dt class="param-name-index-0">SSS</dt><dd class="param-descr-index-0"> a value.</dd></dl>] FullCommentAsXML=[<Function file="{{[^"]+}}overriding-method-comments.mm" line="85" column="6"><Name>foo</Name><USR>c:@F@foo#I#</USR><Abstract><Para> Does something. </Para></Abstract><Parameters><Parameter><Name>SSS</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> a value.</Para></Discussion></Parameter></Parameters></Function>

// CHECK: FullCommentAsHTML=[<p class="para-brief"> Does something. </p><dl><dt class="param-name-index-0">EEE</dt><dd class="param-descr-index-0"> argument to function decl. </dd></dl>] FullCommentAsXML=[<Function file="{{[^"]+}}overriding-method-comments.mm" line="89" column="6"><Name>foo1</Name><USR>c:@F@foo1#I#</USR><Abstract><Para> Does something. </Para></Abstract><Parameters><Parameter><Name>EEE</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> argument to function decl. </Para></Discussion></Parameter></Parameters></Function>

// CHECK: FullCommentAsHTML=[<p class="para-brief"> Does something. </p><dl><dt class="param-name-index-0">TTT</dt><dd class="param-descr-index-0"> argument to function decl. </dd></dl>] FullCommentAsXML=[<Function file="{{[^"]+}}overriding-method-comments.mm" line="91" column="6"><Name>foo1</Name><USR>c:@F@foo1#I#</USR><Abstract><Para> Does something. </Para></Abstract><Parameters><Parameter><Name>TTT</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> argument to function decl. </Para></Discussion></Parameter></Parameters></Function>

/// \brief Documentation
/// \tparam BBB The type, silly.
/// \tparam AAA The type, silly as well.
template<typename AAA, typename BBB>
void foo(AAA, BBB);

template<typename PPP, typename QQQ>
void foo(PPP, QQQ);

// CHECK: FullCommentAsHTML=[<p class="para-brief"> Documentation </p><dl><dt class="tparam-name-index-0">AAA</dt><dd class="tparam-descr-index-0"> The type, silly as well.</dd><dt class="tparam-name-index-1">BBB</dt><dd class="tparam-descr-index-1"> The type, silly. </dd></dl>] FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}overriding-method-comments.mm" line="117" column="6"><Name>foo</Name><USR>c:@FT@&gt;2#T#Tfoo#t0.0#t0.1#</USR><Abstract><Para> Documentation </Para></Abstract><TemplateParameters><Parameter><Name>AAA</Name><Index>0</Index><Discussion><Para> The type, silly as well.</Para></Discussion></Parameter><Parameter><Name>BBB</Name><Index>1</Index><Discussion><Para> The type, silly. </Para></Discussion></Parameter></TemplateParameters></Function>

// CHECK: FullCommentAsHTML=[<p class="para-brief"> Documentation </p><dl><dt class="tparam-name-index-0">PPP</dt><dd class="tparam-descr-index-0"> The type, silly as well.</dd><dt class="tparam-name-index-1">QQQ</dt><dd class="tparam-descr-index-1"> The type, silly. </dd></dl>] FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}overriding-method-comments.mm" line="120" column="6"><Name>foo</Name><USR>c:@FT@&gt;2#T#Tfoo#t0.0#t0.1#</USR><Abstract><Para> Documentation </Para></Abstract><TemplateParameters><Parameter><Name>PPP</Name><Index>0</Index><Discussion><Para> The type, silly as well.</Para></Discussion></Parameter><Parameter><Name>QQQ</Name><Index>1</Index><Discussion><Para> The type, silly. </Para></Discussion></Parameter></TemplateParameters></Function>
