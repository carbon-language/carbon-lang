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
  verifyFormat("return ('aaa') in bbbb;");
}

TEST_F(FormatTestJS, UnderstandsAmpAmp) {
  verifyFormat("e && e.SomeFunction();");
}

TEST_F(FormatTestJS, LiteralOperatorsCanBeKeywords) {
  verifyFormat("not.and.or.not_eq = 1;");
}

TEST_F(FormatTestJS, ES6DestructuringAssignment) {
  verifyFormat("var [a, b, c] = [1, 2, 3];");
  verifyFormat("var {a, b} = {\n"
               "  a: 1,\n"
               "  b: 2\n"
               "};");
}

TEST_F(FormatTestJS, ContainerLiterals) {
  verifyFormat("return {\n"
               "  link: function() {\n"
               "    f();  //\n"
               "  }\n"
               "};");
  verifyFormat("return {\n"
               "  a: a,\n"
               "  link: function() {\n"
               "    f();  //\n"
               "  }\n"
               "};");
  verifyFormat("return {\n"
               "  a: a,\n"
               "  link: function() {\n"
               "    f();  //\n"
               "  },\n"
               "  link: function() {\n"
               "    f();  //\n"
               "  }\n"
               "};");
  verifyFormat("var stuff = {\n"
               "  // comment for update\n"
               "  update: false,\n"
               "  // comment for modules\n"
               "  modules: false,\n"
               "  // comment for tasks\n"
               "  tasks: false\n"
               "};");
  verifyFormat("return {\n"
               "  'finish':\n"
               "      //\n"
               "      a\n"
               "};");
  verifyFormat("var obj = {\n"
               "  fooooooooo: function(x) {\n"
               "    return x.zIsTooLongForOneLineWithTheDeclarationLine();\n"
               "  }\n"
               "};");
}

TEST_F(FormatTestJS, SpacesInContainerLiterals) {
  verifyFormat("var arr = [1, 2, 3];");
  verifyFormat("f({a: 1, b: 2, c: 3});");

  verifyFormat("var object_literal_with_long_name = {\n"
               "  a: 'aaaaaaaaaaaaaaaaaa',\n"
               "  b: 'bbbbbbbbbbbbbbbbbb'\n"
               "};");

  verifyFormat("f({a: 1, b: 2, c: 3});",
               getChromiumStyle(FormatStyle::LK_JavaScript));
  verifyFormat("f({'a': [{}]});");
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

TEST_F(FormatTestJS, GoogModules) {
  verifyFormat("goog.module('this.is.really.absurdly.long');",
               getGoogleJSStyleWithColumns(40));
  verifyFormat("goog.require('this.is.really.absurdly.long');",
               getGoogleJSStyleWithColumns(40));
  verifyFormat("goog.provide('this.is.really.absurdly.long');",
               getGoogleJSStyleWithColumns(40));
  verifyFormat("var long = goog.require('this.is.really.absurdly.long');",
               getGoogleJSStyleWithColumns(40));

  // These should be wrapped normally.
  verifyFormat(
      "var MyLongClassName =\n"
      "    goog.module.get('my.long.module.name.followedBy.MyLongClassName');");
}

TEST_F(FormatTestJS, FormatsFreestandingFunctions) {
  verifyFormat("function outer1(a, b) {\n"
               "  function inner1(a, b) { return a; }\n"
               "  inner1(a, b);\n"
               "}\n"
               "function outer2(a, b) {\n"
               "  function inner2(a, b) { return a; }\n"
               "  inner2(a, b);\n"
               "}");
}

TEST_F(FormatTestJS, FunctionLiterals) {
  verifyFormat("doFoo(function() {});");
  verifyFormat("doFoo(function() { return 1; });");
  verifyFormat("var func = function() {\n"
               "  return 1;\n"
               "};");
  verifyFormat("return {\n"
               "  body: {\n"
               "    setAttribute: function(key, val) { this[key] = val; },\n"
               "    getAttribute: function(key) { return this[key]; },\n"
               "    style: {direction: ''}\n"
               "  }\n"
               "};");
  EXPECT_EQ("abc = xyz ?\n"
            "          function() {\n"
            "            return 1;\n"
            "          } :\n"
            "          function() {\n"
            "            return -1;\n"
            "          };",
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
  verifyFormat("return {\n"
               "  a: 'E',\n"
               "  b: function() {\n"
               "    return function() {\n"
               "      f();  //\n"
               "    };\n"
               "  }\n"
               "};");
  verifyFormat("{\n"
               "  var someVariable = function(x) {\n"
               "    return x.zIsTooLongForOneLineWithTheDeclarationLine();\n"
               "  };\n"
               "}");

  verifyFormat("f({a: function() { return 1; }});",
               getGoogleJSStyleWithColumns(33));
  verifyFormat("f({\n"
               "  a: function() { return 1; }\n"
               "});",
               getGoogleJSStyleWithColumns(32));

  verifyFormat("return {\n"
               "  a: function SomeFunction() {\n"
               "    // ...\n"
               "    return 1;\n"
               "  }\n"
               "};");
  verifyFormat("this.someObject.doSomething(aaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "    .then(goog.bind(function(aaaaaaaaaaa) {\n"
               "      someFunction();\n"
               "      someFunction();\n"
               "    }, this), aaaaaaaaaaaaaaaaa);");

  // FIXME: This is not ideal yet.
  verifyFormat("someFunction(goog.bind(\n"
               "                 function() {\n"
               "                   doSomething();\n"
               "                   doSomething();\n"
               "                 },\n"
               "                 this),\n"
               "             goog.bind(function() {\n"
               "               doSomething();\n"
               "               doSomething();\n"
               "             }, this));");
}

TEST_F(FormatTestJS, InliningFunctionLiterals) {
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_JavaScript);
  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_Inline;
  verifyFormat("var func = function() {\n"
               "  return 1;\n"
               "};",
               Style);
  verifyFormat("var func = doSomething(function() { return 1; });", Style);
  verifyFormat("var outer = function() {\n"
               "  var inner = function() { return 1; }\n"
               "};",
               Style);
  verifyFormat("function outer1(a, b) {\n"
               "  function inner1(a, b) { return a; }\n"
               "}",
               Style);

  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_All;
  verifyFormat("var func = function() { return 1; };", Style);
  verifyFormat("var func = doSomething(function() { return 1; });", Style);
  verifyFormat(
      "var outer = function() { var inner = function() { return 1; } };",
      Style);
  verifyFormat("function outer1(a, b) {\n"
               "  function inner1(a, b) { return a; }\n"
               "}",
               Style);

  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_None;
  verifyFormat("var func = function() {\n"
               "  return 1;\n"
               "};",
               Style);
  verifyFormat("var func = doSomething(function() {\n"
               "  return 1;\n"
               "});",
               Style);
  verifyFormat("var outer = function() {\n"
               "  var inner = function() {\n"
               "    return 1;\n"
               "  }\n"
               "};",
               Style);
  verifyFormat("function outer1(a, b) {\n"
               "  function inner1(a, b) {\n"
               "    return a;\n"
               "  }\n"
               "}",
               Style);
}

TEST_F(FormatTestJS, MultipleFunctionLiterals) {
  verifyFormat("promise.then(\n"
               "    function success() {\n"
               "      doFoo();\n"
               "      doBar();\n"
               "    },\n"
               "    function error() {\n"
               "      doFoo();\n"
               "      doBaz();\n"
               "    },\n"
               "    []);\n");
  verifyFormat("promise.then(\n"
               "    function success() {\n"
               "      doFoo();\n"
               "      doBar();\n"
               "    },\n"
               "    [],\n"
               "    function error() {\n"
               "      doFoo();\n"
               "      doBaz();\n"
               "    });\n");
  // FIXME: Here, we should probably break right after the "(" for consistency.
  verifyFormat("promise.then([],\n"
               "             function success() {\n"
               "               doFoo();\n"
               "               doBar();\n"
               "             },\n"
               "             function error() {\n"
               "               doFoo();\n"
               "               doBaz();\n"
               "             });\n");

  verifyFormat("getSomeLongPromise()\n"
               "    .then(function(value) { body(); })\n"
               "    .thenCatch(function(error) {\n"
               "      body();\n"
               "      body();\n"
               "    });");
  verifyFormat("getSomeLongPromise()\n"
               "    .then(function(value) {\n"
               "      body();\n"
               "      body();\n"
               "    })\n"
               "    .thenCatch(function(error) {\n"
               "      body();\n"
               "      body();\n"
               "    });");

  // FIXME: This is bad, but it used to be formatted correctly by accident.
  verifyFormat("getSomeLongPromise().then(function(value) {\n"
               "  body();\n"
               "}).thenCatch(function(error) { body(); });");
}

TEST_F(FormatTestJS, ReturnStatements) {
  verifyFormat("function() {\n"
               "  return [hello, world];\n"
               "}");
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

  // But, of course, "catch" is a perfectly fine function name in JavaScript.
  verifyFormat("someObject.catch();");
  verifyFormat("someObject.new();");
  verifyFormat("someObject.delete();");
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
  verifyFormat("var regex = /a\\//;\n"
               "var x = 0;");
  EXPECT_EQ("var regex = /\\/*/;\n"
            "var x = 0;",
            format("var regex = /\\/*/;\n"
                   "var x=0;"));
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
  verifyFormat("var regex = /\\xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/;",
               getGoogleJSStyleWithColumns(50));
}

TEST_F(FormatTestJS, RegexLiteralExamples) {
  verifyFormat("var regex = search.match(/(?:\?|&)times=([^?&]+)/i);");
}

TEST_F(FormatTestJS, TypeAnnotations) {
  verifyFormat("var x: string;");
  verifyFormat("function x(): string {\n  return 'x';\n}");
  verifyFormat("function x(y: string): string {\n  return 'x';\n}");
  verifyFormat("for (var y: string in x) {\n  x();\n}");
  verifyFormat("((a: string, b: number): string => a + b);");
  verifyFormat("var x: (y: number) => string;");
  verifyFormat("var x: P<string, (a: number) => string>;");
}

TEST_F(FormatTestJS, ClassDeclarations) {
  verifyFormat("class C {\n  x: string = 12;\n}");
  verifyFormat("class C {\n  x(): string => 12;\n}");
  verifyFormat("class C {\n  ['x' + 2]: string = 12;\n}");
  verifyFormat("class C {\n  private x: string = 12;\n}");
  verifyFormat("class C {\n  private static x: string = 12;\n}");
  verifyFormat("class C {\n  static x(): string { return 'asd'; }\n}");
  verifyFormat("class C extends P implements I {}");
}

TEST_F(FormatTestJS, MetadataAnnotations) {
  verifyFormat("@A\nclass C {\n}");
  verifyFormat("@A({arg: 'value'})\nclass C {\n}");
  verifyFormat("@A\n@B\nclass C {\n}");
  verifyFormat("class C {\n  @A x: string;\n}");
  verifyFormat("class C {\n"
               "  @A\n"
               "  private x(): string {\n"
               "    return 'y';\n"
               "  }\n"
               "}");
  verifyFormat("class X {}\n"
               "class Y {}");
}

TEST_F(FormatTestJS, Modules) {
  verifyFormat("import SomeThing from 'some/module.js';");
  verifyFormat("import {X, Y} from 'some/module.js';");
  verifyFormat("import {\n"
               "  VeryLongImportsAreAnnoying,\n"
               "  VeryLongImportsAreAnnoying,\n"
               "  VeryLongImportsAreAnnoying,\n"
               "  VeryLongImportsAreAnnoying\n"
               "} from 'some/module.js';");
  verifyFormat("import {\n"
               "  X,\n"
               "  Y,\n"
               "} from 'some/module.js';");
  verifyFormat("import {\n"
               "  X,\n"
               "  Y,\n"
               "} from 'some/long/module.js';",
               getGoogleJSStyleWithColumns(20));
  verifyFormat("import {X as myLocalX, Y as myLocalY} from 'some/module.js';");
  verifyFormat("import * as lib from 'some/module.js';");
  verifyFormat("var x = {\n  import: 1\n};\nx.import = 2;");

  verifyFormat("export function fn() {\n"
               "  return 'fn';\n"
               "}");
  verifyFormat("export const x = 12;");
  verifyFormat("export default class X {}");
  verifyFormat("export {X, Y} from 'some/module.js';");
  verifyFormat("export {\n"
               "  X,\n"
               "  Y,\n"
               "} from 'some/module.js';");
  verifyFormat("export class C {\n"
               "  x: number;\n"
               "  y: string;\n"
               "}");
  verifyFormat("export class X { y: number; }");
  verifyFormat("export default class X { y: number }");
  verifyFormat("export default function() {\n  return 1;\n}");
  verifyFormat("export var x = 12;");
  verifyFormat("export var x: number = 12;");
  verifyFormat("export const y = {\n"
               "  a: 1,\n"
               "  b: 2\n"
               "};");
}

TEST_F(FormatTestJS, TemplateStrings) {
  // Keeps any whitespace/indentation within the template string.
  EXPECT_EQ("var x = `hello\n"
            "     ${  name    }\n"
            "  !`;",
            format("var x    =    `hello\n"
                   "     ${  name    }\n"
                   "  !`;"));

  // FIXME: +1 / -1 offsets are to work around clang-format miscalculating
  // widths for unknown tokens that are not whitespace (e.g. '`'). Remove when
  // the code is corrected.

  verifyFormat("var x =\n"
               "    `hello ${world}` >= some();",
               getGoogleJSStyleWithColumns(34)); // Barely doesn't fit.
  verifyFormat("var x = `hello ${world}` >= some();",
               getGoogleJSStyleWithColumns(35 + 1)); // Barely fits.
  EXPECT_EQ("var x = `hello\n"
            "  ${world}` >=\n"
            "        some();",
            format("var x =\n"
                   "    `hello\n"
                   "  ${world}` >= some();",
                   getGoogleJSStyleWithColumns(21))); // Barely doesn't fit.
  EXPECT_EQ("var x = `hello\n"
            "  ${world}` >= some();",
            format("var x =\n"
                   "    `hello\n"
                   "  ${world}` >= some();",
                   getGoogleJSStyleWithColumns(22))); // Barely fits.

  verifyFormat("var x =\n    `h`;", getGoogleJSStyleWithColumns(13 - 1));
  EXPECT_EQ(
      "var x =\n    `multi\n  line`;",
      format("var x = `multi\n  line`;", getGoogleJSStyleWithColumns(14 - 1)));

  // Two template strings.
  verifyFormat("var x = `hello` == `hello`;");
}

} // end namespace tooling
} // end namespace clang
