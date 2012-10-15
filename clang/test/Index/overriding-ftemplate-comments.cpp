// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng %s > %t/out
// RUN: FileCheck %s < %t/out
// Test to search overridden methods for documentation when overriding method has none. rdar://12378793

// Ensure that XML we generate is not invalid.
// RUN: FileCheck %s -check-prefix=WRONG < %t/out
// WRONG-NOT: CommentXMLInvalid

/// \tparam
/// \param AAA Blah blah
template<typename T>
void comment_to_html_conversion_17(T AAA);

template<typename T>
void comment_to_html_conversion_17(T PPP);

/// \tparam BBB Bbb
/// \tparam AAA Aaa
template<typename AAA, typename BBB>
void comment_to_html_conversion_19(AAA aaa, BBB bbb);

template<typename PPP, typename QQQ>
void comment_to_html_conversion_19(PPP aaa, QQQ bbb);

/// \tparam BBB Bbb
/// \tparam UUU Zzz
/// \tparam CCC Ccc
/// \tparam AAA Aaa
template<typename AAA, typename BBB, int CCC>
void comment_to_html_conversion_20(AAA aaa, BBB bbb);

template<typename PPP, typename QQQ, int RRR>
void comment_to_html_conversion_20(PPP aaa, QQQ bbb);

/// \tparam AAA Aaa
/// \tparam BBB Bbb
/// \tparam CCC Ccc
/// \tparam DDD Ddd
template<template<template<typename CCC> class DDD, class BBB> class AAA>
void comment_to_html_conversion_21();

template<template<template<typename RRR> class SSS, class QQQ> class PPP>
void comment_to_html_conversion_21();

/// \tparam C1 Ccc 1
/// \tparam AAA Zzz
/// \tparam C2 Ccc 2
/// \tparam C3 Ccc 3
/// \tparam C4 Ccc 4
/// \tparam BBB Bbb
template<class C1, template<class C2, template<class C3, class C4> class BBB> class AAA>
void comment_to_html_conversion_22();


template<class CCC1, template<class CCC2, template<class CCC3, class CCC4> class QQQ> class PPP>
void comment_to_html_conversion_22();

// CHECK: FunctionTemplate=comment_to_html_conversion_17:14:6 RawComment=[/// \tparam\n/// \param AAA Blah blah] RawCommentRange=[11:1 - 12:25] FullCommentAsHTML=[<dl><dt class="param-name-index-0">AAA</dt><dd class="param-descr-index-0"> Blah blah</dd></dl>] FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}overriding-ftemplate-comments.cpp" line="14" column="6"><Name>comment_to_html_conversion_17</Name><USR>c:@FT@&gt;1#Tcomment_to_html_conversion_17#t0.0#</USR><Parameters><Parameter><Name>AAA</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Blah blah</Para></Discussion></Parameter></Parameters></Function>

// CHECK: FunctionTemplate=comment_to_html_conversion_17:17:6 RawComment=[/// \tparam\n/// \param AAA Blah blah] RawCommentRange=[11:1 - 12:25] FullCommentAsHTML=[<dl><dt class="param-name-index-0">PPP</dt><dd class="param-descr-index-0"> Blah blah</dd></dl>] FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}overriding-ftemplate-comments.cpp" line="14" column="6"><Name>comment_to_html_conversion_17</Name><USR>c:@FT@&gt;1#Tcomment_to_html_conversion_17#t0.0#</USR><Parameters><Parameter><Name>PPP</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Blah blah</Para></Discussion></Parameter></Parameters></Function>

// CHECK: FunctionTemplate=comment_to_html_conversion_19:22:6 RawComment=[/// \tparam BBB Bbb\n/// \tparam AAA Aaa] RawCommentRange=[19:1 - 20:20] FullCommentAsHTML=[<dl><dt class="tparam-name-index-0">AAA</dt><dd class="tparam-descr-index-0"> Aaa</dd><dt class="tparam-name-index-1">BBB</dt><dd class="tparam-descr-index-1"> Bbb </dd></dl>] FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}overriding-ftemplate-comments.cpp" line="22" column="6"><Name>comment_to_html_conversion_19</Name><USR>c:@FT@&gt;2#T#Tcomment_to_html_conversion_19#t0.0#t0.1#</USR><TemplateParameters><Parameter><Name>AAA</Name><Index>0</Index><Discussion><Para> Aaa</Para></Discussion></Parameter><Parameter><Name>BBB</Name><Index>1</Index><Discussion><Para> Bbb </Para></Discussion></Parameter></TemplateParameters></Function>

// CHECK: FunctionTemplate=comment_to_html_conversion_19:25:6 RawComment=[/// \tparam BBB Bbb\n/// \tparam AAA Aaa] RawCommentRange=[19:1 - 20:20] FullCommentAsHTML=[<dl><dt class="tparam-name-index-0">PPP</dt><dd class="tparam-descr-index-0"> Aaa</dd><dt class="tparam-name-index-1">QQQ</dt><dd class="tparam-descr-index-1"> Bbb </dd></dl>] FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}overriding-ftemplate-comments.cpp" line="22" column="6"><Name>comment_to_html_conversion_19</Name><USR>c:@FT@&gt;2#T#Tcomment_to_html_conversion_19#t0.0#t0.1#</USR><TemplateParameters><Parameter><Name>PPP</Name><Index>0</Index><Discussion><Para> Aaa</Para></Discussion></Parameter><Parameter><Name>QQQ</Name><Index>1</Index><Discussion><Para> Bbb </Para></Discussion></Parameter></TemplateParameters></Function>

// CHECK: FunctionTemplate=comment_to_html_conversion_20:32:6 RawComment=[/// \tparam BBB Bbb\n/// \tparam UUU Zzz\n/// \tparam CCC Ccc\n/// \tparam AAA Aaa] RawCommentRange=[27:1 - 30:20] FullCommentAsHTML=[<dl><dt class="tparam-name-index-0">AAA</dt><dd class="tparam-descr-index-0"> Aaa</dd><dt class="tparam-name-index-1">BBB</dt><dd class="tparam-descr-index-1"> Bbb </dd><dt class="tparam-name-index-2">CCC</dt><dd class="tparam-descr-index-2"> Ccc </dd><dt class="tparam-name-index-invalid">UUU</dt><dd class="tparam-descr-index-invalid"> Zzz </dd></dl>] FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}overriding-ftemplate-comments.cpp" line="32" column="6"><Name>comment_to_html_conversion_20</Name><USR>c:@FT@&gt;3#T#T#NIcomment_to_html_conversion_20#t0.0#t0.1#</USR><TemplateParameters><Parameter><Name>AAA</Name><Index>0</Index><Discussion><Para> Aaa</Para></Discussion></Parameter><Parameter><Name>BBB</Name><Index>1</Index><Discussion><Para> Bbb </Para></Discussion></Parameter><Parameter><Name>CCC</Name><Index>2</Index><Discussion><Para> Ccc </Para></Discussion></Parameter><Parameter><Name>UUU</Name><Discussion><Para> Zzz </Para></Discussion></Parameter></TemplateParameters></Function>

// CHECK: FunctionTemplate=comment_to_html_conversion_20:35:6 RawComment=[/// \tparam BBB Bbb\n/// \tparam UUU Zzz\n/// \tparam CCC Ccc\n/// \tparam AAA Aaa] RawCommentRange=[27:1 - 30:20] FullCommentAsHTML=[<dl><dt class="tparam-name-index-0">PPP</dt><dd class="tparam-descr-index-0"> Aaa</dd><dt class="tparam-name-index-1">QQQ</dt><dd class="tparam-descr-index-1"> Bbb </dd><dt class="tparam-name-index-2">RRR</dt><dd class="tparam-descr-index-2"> Ccc </dd><dt class="tparam-name-index-invalid">UUU</dt><dd class="tparam-descr-index-invalid"> Zzz </dd></dl>] FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}overriding-ftemplate-comments.cpp" line="32" column="6"><Name>comment_to_html_conversion_20</Name><USR>c:@FT@&gt;3#T#T#NIcomment_to_html_conversion_20#t0.0#t0.1#</USR><TemplateParameters><Parameter><Name>PPP</Name><Index>0</Index><Discussion><Para> Aaa</Para></Discussion></Parameter><Parameter><Name>QQQ</Name><Index>1</Index><Discussion><Para> Bbb </Para></Discussion></Parameter><Parameter><Name>RRR</Name><Index>2</Index><Discussion><Para> Ccc </Para></Discussion></Parameter><Parameter><Name>UUU</Name><Discussion><Para> Zzz </Para></Discussion></Parameter></TemplateParameters></Function>

// CHECK: FunctionTemplate=comment_to_html_conversion_21:42:6 RawComment=[/// \tparam AAA Aaa\n/// \tparam BBB Bbb\n/// \tparam CCC Ccc\n/// \tparam DDD Ddd] RawCommentRange=[37:1 - 40:20] FullCommentAsHTML=[<dl><dt class="tparam-name-index-0">AAA</dt><dd class="tparam-descr-index-0"> Aaa </dd><dt class="tparam-name-index-other">BBB</dt><dd class="tparam-descr-index-other"> Bbb </dd><dt class="tparam-name-index-other">CCC</dt><dd class="tparam-descr-index-other"> Ccc </dd><dt class="tparam-name-index-other">DDD</dt><dd class="tparam-descr-index-other"> Ddd</dd></dl>] FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}overriding-ftemplate-comments.cpp" line="42" column="6"><Name>comment_to_html_conversion_21</Name><USR>c:@FT@&gt;1#t&gt;2#t&gt;1#T#Tcomment_to_html_conversion_21#</USR><TemplateParameters><Parameter><Name>AAA</Name><Index>0</Index><Discussion><Para> Aaa </Para></Discussion></Parameter><Parameter><Name>BBB</Name><Discussion><Para> Bbb </Para></Discussion></Parameter><Parameter><Name>CCC</Name><Discussion><Para> Ccc </Para></Discussion></Parameter><Parameter><Name>DDD</Name><Discussion><Para> Ddd</Para></Discussion></Parameter></TemplateParameters></Function>

// CHECK: FunctionTemplate=comment_to_html_conversion_21:45:6 RawComment=[/// \tparam AAA Aaa\n/// \tparam BBB Bbb\n/// \tparam CCC Ccc\n/// \tparam DDD Ddd] RawCommentRange=[37:1 - 40:20] FullCommentAsHTML=[<dl><dt class="tparam-name-index-0">PPP</dt><dd class="tparam-descr-index-0"> Aaa </dd><dt class="tparam-name-index-other">QQQ</dt><dd class="tparam-descr-index-other"> Bbb </dd><dt class="tparam-name-index-other">RRR</dt><dd class="tparam-descr-index-other"> Ccc </dd><dt class="tparam-name-index-other">SSS</dt><dd class="tparam-descr-index-other"> Ddd</dd></dl>] FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}overriding-ftemplate-comments.cpp" line="42" column="6"><Name>comment_to_html_conversion_21</Name><USR>c:@FT@&gt;1#t&gt;2#t&gt;1#T#Tcomment_to_html_conversion_21#</USR><TemplateParameters><Parameter><Name>PPP</Name><Index>0</Index><Discussion><Para> Aaa </Para></Discussion></Parameter><Parameter><Name>QQQ</Name><Discussion><Para> Bbb </Para></Discussion></Parameter><Parameter><Name>RRR</Name><Discussion><Para> Ccc </Para></Discussion></Parameter><Parameter><Name>SSS</Name><Discussion><Para> Ddd</Para></Discussion></Parameter></TemplateParameters></Function>

// CHECK: FunctionTemplate=comment_to_html_conversion_22:54:6 RawComment=[/// \tparam C1 Ccc 1\n/// \tparam AAA Zzz\n/// \tparam C2 Ccc 2\n/// \tparam C3 Ccc 3\n/// \tparam C4 Ccc 4\n/// \tparam BBB Bbb] RawCommentRange=[47:1 - 52:20] FullCommentAsHTML=[<dl><dt class="tparam-name-index-0">C1</dt><dd class="tparam-descr-index-0"> Ccc 1 </dd><dt class="tparam-name-index-1">AAA</dt><dd class="tparam-descr-index-1"> Zzz </dd><dt class="tparam-name-index-other">C2</dt><dd class="tparam-descr-index-other"> Ccc 2 </dd><dt class="tparam-name-index-other">C3</dt><dd class="tparam-descr-index-other"> Ccc 3 </dd><dt class="tparam-name-index-other">C4</dt><dd class="tparam-descr-index-other"> Ccc 4 </dd><dt class="tparam-name-index-other">BBB</dt><dd class="tparam-descr-index-other"> Bbb</dd></dl>] FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}overriding-ftemplate-comments.cpp" line="54" column="6"><Name>comment_to_html_conversion_22</Name><USR>c:@FT@&gt;2#T#t&gt;2#T#t&gt;2#T#Tcomment_to_html_conversion_22#</USR><TemplateParameters><Parameter><Name>C1</Name><Index>0</Index><Discussion><Para> Ccc 1 </Para></Discussion></Parameter><Parameter><Name>AAA</Name><Index>1</Index><Discussion><Para> Zzz </Para></Discussion></Parameter><Parameter><Name>C2</Name><Discussion><Para> Ccc 2 </Para></Discussion></Parameter><Parameter><Name>C3</Name><Discussion><Para> Ccc 3 </Para></Discussion></Parameter><Parameter><Name>C4</Name><Discussion><Para> Ccc 4 </Para></Discussion></Parameter><Parameter><Name>BBB</Name><Discussion><Para> Bbb</Para></Discussion></Parameter></TemplateParameters></Function>

// CHECK: FunctionTemplate=comment_to_html_conversion_22:58:6 RawComment=[/// \tparam C1 Ccc 1\n/// \tparam AAA Zzz\n/// \tparam C2 Ccc 2\n/// \tparam C3 Ccc 3\n/// \tparam C4 Ccc 4\n/// \tparam BBB Bbb] RawCommentRange=[47:1 - 52:20] FullCommentAsHTML=[<dl><dt class="tparam-name-index-0">CCC1</dt><dd class="tparam-descr-index-0"> Ccc 1 </dd><dt class="tparam-name-index-1">PPP</dt><dd class="tparam-descr-index-1"> Zzz </dd><dt class="tparam-name-index-other">CCC2</dt><dd class="tparam-descr-index-other"> Ccc 2 </dd><dt class="tparam-name-index-other">CCC3</dt><dd class="tparam-descr-index-other"> Ccc 3 </dd><dt class="tparam-name-index-other">CCC4</dt><dd class="tparam-descr-index-other"> Ccc 4 </dd><dt class="tparam-name-index-other">QQQ</dt><dd class="tparam-descr-index-other"> Bbb</dd></dl>] FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}overriding-ftemplate-comments.cpp" line="54" column="6"><Name>comment_to_html_conversion_22</Name><USR>c:@FT@&gt;2#T#t&gt;2#T#t&gt;2#T#Tcomment_to_html_conversion_22#</USR><TemplateParameters><Parameter><Name>CCC1</Name><Index>0</Index><Discussion><Para> Ccc 1 </Para></Discussion></Parameter><Parameter><Name>PPP</Name><Index>1</Index><Discussion><Para> Zzz </Para></Discussion></Parameter><Parameter><Name>CCC2</Name><Discussion><Para> Ccc 2 </Para></Discussion></Parameter><Parameter><Name>CCC3</Name><Discussion><Para> Ccc 3 </Para></Discussion></Parameter><Parameter><Name>CCC4</Name><Discussion><Para> Ccc 4 </Para></Discussion></Parameter><Parameter><Name>QQQ</Name><Discussion><Para> Bbb</Para></Discussion></Parameter></TemplateParameters></Function>
