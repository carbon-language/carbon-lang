//===- unittest/Format/FormatTestJS.cpp - Formatting unit tests for JS ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FormatTestUtils.h"
#include "clang/Format/Format.h"
#include "llvm/Support/Debug.h"
#include "gtest/gtest.h"

#define DEBUG_TYPE "format-test"

namespace clang {
namespace format {

class FormatTestJS : public ::testing::Test {
protected:
  static std::string format(llvm::StringRef Code, unsigned Offset,
                            unsigned Length, const FormatStyle &Style) {
    DEBUG(llvm::errs() << "---\n");
    DEBUG(llvm::errs() << Code << "\n\n");
    std::vector<tooling::Range> Ranges(1, tooling::Range(Offset, Length));
    tooling::Replacements Replaces = reformat(Style, Code, Ranges);
    std::string Result = applyAllReplacements(Code, Replaces);
    EXPECT_NE("", Result);
    DEBUG(llvm::errs() << "\n" << Result << "\n\n");
    return Result;
  }

  static std::string format(
      llvm::StringRef Code,
      const FormatStyle &Style = getGoogleStyle(FormatStyle::LK_JavaScript)) {
    return format(Code, 0, Code.size(), Style);
  }

  static FormatStyle getGoogleJSStyleWithColumns(unsigned ColumnLimit) {
    FormatStyle Style = getGoogleStyle(FormatStyle::LK_JavaScript);
    Style.ColumnLimit = ColumnLimit;
    return Style;
  }

  static void verifyFormat(
      llvm::StringRef Code,
      const FormatStyle &Style = getGoogleStyle(FormatStyle::LK_JavaScript)) {
    EXPECT_EQ(Code.str(), format(test::messUp(Code), Style));
  }
};

TEST_F(FormatTestJS, UnderstandsJavaScriptOperators) {
  verifyFormat("a == = b;");
  verifyFormat("a != = b;");

  verifyFormat("a === b;");
  verifyFormat("aaaaaaa ===\n    b;", getGoogleJSStyleWithColumns(10));
  verifyFormat("a !== b;");
  verifyFormat("aaaaaaa !==\n    b;", getGoogleJSStyleWithColumns(10));
  verifyFormat("if (a + b + c +\n"
               "        d !==\n"
               "    e + f + g)\n"
               "  q();",
               getGoogleJSStyleWithColumns(20));

  verifyFormat("a >> >= b;");

  verifyFormat("a >>> b;");
  verifyFormat("aaaaaaa >>>\n    b;", getGoogleJSStyleWithColumns(10));
  verifyFormat("a >>>= b;");
  verifyFormat("aaaaaaa >>>=\n    b;", getGoogleJSStyleWithColumns(10));
  verifyFormat("if (a + b + c +\n"
               "        d >>>\n"
               "    e + f + g)\n"
               "  q();",
               getGoogleJSStyleWithColumns(20));
  verifyFormat("var x = aaaaaaaaaa ?\n"
               "            bbbbbb :\n"
               "            ccc;",
               getGoogleJSStyleWithColumns(20));

  verifyFormat("var b = a.map((x) => x + 1);");
}

TEST_F(FormatTestJS, ES6DestructuringAssignment) {
  verifyFormat("var [a, b, c] = [1, 2, 3];");
  verifyFormat("var {a, b} = {a: 1, b: 2};");
}

TEST_F(FormatTestJS, SpacesInContainerLiterals) {
  verifyFormat("var arr = [1, 2, 3];");
  verifyFormat("var obj = {a: 1, b: 2, c: 3};");

  verifyFormat("var object_literal_with_long_name = {\n"
               "  a: 'aaaaaaaaaaaaaaaaaa',\n"
               "  b: 'bbbbbbbbbbbbbbbbbb'\n"
               "};");

  verifyFormat("var obj = {a: 1, b: 2, c: 3};",
               getChromiumStyle(FormatStyle::LK_JavaScript));
  verifyFormat("someVariable = {'a': [{}]};");
}

TEST_F(FormatTestJS, SingleQuoteStrings) {
  verifyFormat("this.function('', true);");
}

TEST_F(FormatTestJS, GoogScopes) {
  verifyFormat("goog.scope(function() {\n"
               "var x = a.b;\n"
               "var y = c.d;\n"
               "});  // goog.scope");
}

TEST_F(FormatTestJS, Closures) {
  verifyFormat("doFoo(function() { return 1; });");
  verifyFormat("var func = function() { return 1; };");
  verifyFormat("return {\n"
               "  body: {\n"
               "    setAttribute: function(key, val) { this[key] = val; },\n"
               "    getAttribute: function(key) { return this[key]; },\n"
               "    style: {direction: ''}\n"
               "  }\n"
               "};");
  EXPECT_EQ("abc = xyz ? function() { return 1; } : function() { return -1; };",
            format("abc=xyz?function(){return 1;}:function(){return -1;};"));

  verifyFormat("var closure = goog.bind(\n"
               "    function() {  // comment\n"
               "      foo();\n"
               "      bar();\n"
               "    },\n"
               "    this, arg1IsReallyLongAndNeeedsLineBreaks,\n"
               "    arg3IsReallyLongAndNeeedsLineBreaks);");
  verifyFormat("var closure = goog.bind(function() {  // comment\n"
               "  foo();\n"
               "  bar();\n"
               "}, this);");

  verifyFormat("var x = {a: function() { return 1; }};",
               getGoogleJSStyleWithColumns(38));
  verifyFormat("var x = {\n"
               "  a: function() { return 1; }\n"
               "};",
               getGoogleJSStyleWithColumns(37));
}

TEST_F(FormatTestJS, ReturnStatements) {
  verifyFormat("function() { return [hello, world]; }");
}

TEST_F(FormatTestJS, ClosureStyleComments) {
  verifyFormat("var x = /** @type {foo} */ (bar);");
}

TEST_F(FormatTestJS, TryCatch) {
  verifyFormat("try {\n"
               "  f();\n"
               "} catch (e) {\n"
               "  g();\n"
               "} finally {\n"
               "  h();\n"
               "}");
}

TEST_F(FormatTestJS, StringLiteralConcatenation) {
  verifyFormat("var literal = 'hello ' +\n"
               "              'world';");
}

TEST_F(FormatTestJS, RegexLiteralClassification) {
  // Regex literals.
  verifyFormat("var regex = /abc/;");
  verifyFormat("f(/abc/);");
  verifyFormat("f(abc, /abc/);");
  verifyFormat("some_map[/abc/];");
  verifyFormat("var x = a ? /abc/ : /abc/;");
  verifyFormat("for (var i = 0; /abc/.test(s[i]); i++) {\n}");
  verifyFormat("var x = !/abc/.test(y);");
  verifyFormat("var x = a && /abc/.test(y);");
  verifyFormat("var x = a || /abc/.test(y);");
  verifyFormat("var x = a + /abc/.search(y);");
  verifyFormat("var regexs = {/abc/, /abc/};");
  verifyFormat("return /abc/;");

  // Not regex literals.
  verifyFormat("var a = a / 2 + b / 3;");
}

TEST_F(FormatTestJS, RegexLiteralSpecialCharacters) {
  verifyFormat("var regex = /a*/;");
  verifyFormat("var regex = /a+/;");
  verifyFormat("var regex = /a?/;");
  verifyFormat("var regex = /.a./;");
  verifyFormat("var regex = /a\\*/;");
  verifyFormat("var regex = /^a$/;");
  verifyFormat("var regex = /\\/a/;");
  verifyFormat("var regex = /(?:x)/;");
  verifyFormat("var regex = /x(?=y)/;");
  verifyFormat("var regex = /x(?!y)/;");
  verifyFormat("var regex = /x|y/;");
  verifyFormat("var regex = /a{2}/;");
  verifyFormat("var regex = /a{1,3}/;");
  verifyFormat("var regex = /[abc]/;");
  verifyFormat("var regex = /[^abc]/;");
  verifyFormat("var regex = /[\\b]/;");
  verifyFormat("var regex = /\\b/;");
  verifyFormat("var regex = /\\B/;");
  verifyFormat("var regex = /\\d/;");
  verifyFormat("var regex = /\\D/;");
  verifyFormat("var regex = /\\f/;");
  verifyFormat("var regex = /\\n/;");
  verifyFormat("var regex = /\\r/;");
  verifyFormat("var regex = /\\s/;");
  verifyFormat("var regex = /\\S/;");
  verifyFormat("var regex = /\\t/;");
  verifyFormat("var regex = /\\v/;");
  verifyFormat("var regex = /\\w/;");
  verifyFormat("var regex = /\\W/;");
  verifyFormat("var regex = /a(a)\\1/;");
  verifyFormat("var regex = /\\0/;");
  verifyFormat("var regex = /\\\\/g;");
  verifyFormat("var regex = /\\a\\\\/g;");
  verifyFormat("var regex = /\a\\//g;");
}

TEST_F(FormatTestJS, RegexLiteralModifiers) {
  verifyFormat("var regex = /abc/g;");
  verifyFormat("var regex = /abc/i;");
  verifyFormat("var regex = /abc/m;");
  verifyFormat("var regex = /abc/y;");
}

TEST_F(FormatTestJS, RegexLiteralLength) {
  verifyFormat("var regex = /aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/;",
               getGoogleJSStyleWithColumns(60));
  verifyFormat("var regex =\n"
               "    /aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/;",
               getGoogleJSStyleWithColumns(60));
}

TEST_F(FormatTestJS, RegexLiteralExamples) {
  verifyFormat("var regex = search.match(/(?:\?|&)times=([^?&]+)/i);");
}

} // end namespace tooling
} // end namespace clang
