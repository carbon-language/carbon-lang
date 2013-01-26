// RUN: %clang_cc1 -triple x86_64-pc-linux -std=c++11 -ast-dump -fcolor-diagnostics %s | FileCheck --strict-whitespace %s
// REQUIRES: ansi-escape-sequences

/// <a>Hello</a>
/// <br/>
int Test __attribute__((unused));

/// Comment
void TestAttributedStmt() {
  switch (1) {
  case 1:
    [[clang::fallthrough]];
  case 2:
    ;
  }
}

class __attribute__((lockable)) Mutex {
  /// A variable
  int var1;
  /// Another variable
  ///
  /// Like the other variable, but different
  int var2;
} mu1, mu2;
int TestExpr __attribute__((guarded_by(mu1)));

//CHECK: {{^}}([[GREEN:.\[0;1;32m]]TranslationUnitDecl[[RESET:.\[0m]][[Yellow:.\[0;33m]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]<invalid sloc>[[RESET]]>{{$}}
//CHECK: {{^}}  ([[GREEN]]TypedefDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]<invalid sloc>[[RESET]]>[[CYAN:.\[0;1;36m]] __int128_t[[RESET]] [[Green:.\[0;32m]]'__int128'[[RESET]]){{$}}
//CHECK: {{^}}  ([[GREEN]]TypedefDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]<invalid sloc>[[RESET]]>[[CYAN]] __uint128_t[[RESET]] [[Green]]'unsigned __int128'[[RESET]]){{$}}
//CHECK: {{^}}  ([[GREEN]]TypedefDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]<invalid sloc>[[RESET]]>[[CYAN]] __builtin_va_list[[RESET]] [[Green]]'__va_list_tag [1]'[[RESET]]){{$}}
//CHECK: {{^}}  ([[GREEN]]VarDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]{{.*}}ast-dump-color.cpp:6:1[[RESET]], [[Yellow]]col:5[[RESET]]>[[CYAN]] Test[[RESET]] [[Green]]'int'[[RESET]]{{$}}
//CHECK: {{^}}    ([[BLUE:.\[0;1;34m]]UnusedAttr[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:25[[RESET]]>){{$}}
//CHECK: {{^}}    ([[YELLOW:.\[0;1;33m]]FullComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:4:4[[RESET]], [[Yellow]]line:5:8[[RESET]]>{{$}}
//CHECK: {{^}}      ([[YELLOW]]ParagraphComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:4:4[[RESET]], [[Yellow]]line:5:8[[RESET]]>{{$}}
//CHECK: {{^}}        ([[YELLOW]]TextComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:4:4[[RESET]]> Text=" "){{$}}
//CHECK: {{^}}        ([[YELLOW]]HTMLStartTagComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:5[[RESET]], [[Yellow]]col:7[[RESET]]> Name="a"){{$}}
//CHECK: {{^}}        ([[YELLOW]]TextComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:8[[RESET]], [[Yellow]]col:12[[RESET]]> Text="Hello"){{$}}
//CHECK: {{^}}        ([[YELLOW]]HTMLEndTagComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:13[[RESET]], [[Yellow]]col:16[[RESET]]> Name="a"){{$}}
//CHECK: {{^}}        ([[YELLOW]]TextComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:5:4[[RESET]]> Text=" "){{$}}
//CHECK: {{^}}        ([[YELLOW]]HTMLStartTagComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:5[[RESET]], [[Yellow]]col:8[[RESET]]> Name="br" SelfClosing)))){{$}}
//CHECK: {{^}}  ([[GREEN]]FunctionDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:9:1[[RESET]], [[Yellow]]line:16:1[[RESET]]>[[CYAN]] TestAttributedStmt[[RESET]] [[Green]]'void (void)'[[RESET]]{{$}}
//CHECK: {{^}}    ([[MAGENTA:.\[0;1;35m]]CompoundStmt[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:9:27[[RESET]], [[Yellow]]line:16:1[[RESET]]>{{$}}
//CHECK: {{^}}      ([[MAGENTA]]SwitchStmt[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:10:3[[RESET]], [[Yellow]]line:15:3[[RESET]]>{{$}}
//CHECK: {{^}}        ([[Blue:.\[0;34m]]<<<NULL>>>[[RESET]]){{$}}
//CHECK: {{^}}        ([[MAGENTA]]IntegerLiteral[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:10:11[[RESET]]> [[Green]]'int'[[RESET]][[Cyan:.\[0;36m]][[RESET]][[Cyan]][[RESET]][[CYAN]] 1[[RESET]]){{$}}
//CHECK: {{^}}        ([[MAGENTA]]CompoundStmt[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:14[[RESET]], [[Yellow]]line:15:3[[RESET]]>{{$}}
//CHECK: {{^}}          ([[MAGENTA]]CaseStmt[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:11:3[[RESET]], [[Yellow]]line:12:27[[RESET]]>{{$}}
//CHECK: {{^}}            ([[MAGENTA]]IntegerLiteral[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:11:8[[RESET]]> [[Green]]'int'[[RESET]][[Cyan]][[RESET]][[Cyan]][[RESET]][[CYAN]] 1[[RESET]]){{$}}
//CHECK: {{^}}            ([[Blue]]<<<NULL>>>[[RESET]]){{$}}
//CHECK: {{^}}            ([[MAGENTA]]AttributedStmt[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:12:5[[RESET]], [[Yellow]]col:27[[RESET]]>{{$}}
//CHECK: {{^}}              ([[BLUE]]FallThroughAttr[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:7[[RESET]], [[Yellow]]col:14[[RESET]]>){{$}}
//CHECK: {{^}}              ([[MAGENTA]]NullStmt[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:27[[RESET]]>))){{$}}
//CHECK: {{^}}          ([[MAGENTA]]CaseStmt[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:13:3[[RESET]], [[Yellow]]line:14:5[[RESET]]>{{$}}
//CHECK: {{^}}            ([[MAGENTA]]IntegerLiteral[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:13:8[[RESET]]> [[Green]]'int'[[RESET]][[Cyan]][[RESET]][[Cyan]][[RESET]][[CYAN]] 2[[RESET]]){{$}}
//CHECK: {{^}}            ([[Blue]]<<<NULL>>>[[RESET]]){{$}}
//CHECK: {{^}}            ([[MAGENTA]]NullStmt[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:14:5[[RESET]]>))))){{$}}
//CHECK: {{^}}    ([[YELLOW]]FullComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:8:4[[RESET]], [[Yellow]]col:11[[RESET]]>{{$}}
//CHECK: {{^}}      ([[YELLOW]]ParagraphComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:4[[RESET]], [[Yellow]]col:11[[RESET]]>{{$}}
//CHECK: {{^}}        ([[YELLOW]]TextComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:4[[RESET]], [[Yellow]]col:11[[RESET]]> Text=" Comment")))){{$}}
//CHECK: {{^}}  ([[GREEN]]CXXRecordDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:18:1[[RESET]], [[Yellow]]line:25:1[[RESET]]> class[[CYAN]] Mutex[[RESET]]{{$}}
//CHECK: {{^}}    ([[BLUE]]LockableAttr[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:18:22[[RESET]]>){{$}}
//CHECK: {{^}}    ([[GREEN]]CXXRecordDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:1[[RESET]], [[Yellow]]col:33[[RESET]]> class[[CYAN]] Mutex[[RESET]]){{$}}
//CHECK: {{^}}    ([[GREEN]]FieldDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:20:3[[RESET]], [[Yellow]]col:7[[RESET]]>[[CYAN]] var1[[RESET]] [[Green]]'int'[[RESET]]{{$}}
//CHECK: {{^}}      ([[YELLOW]]FullComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:19:6[[RESET]], [[Yellow]]col:16[[RESET]]>{{$}}
//CHECK: {{^}}        ([[YELLOW]]ParagraphComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:6[[RESET]], [[Yellow]]col:16[[RESET]]>{{$}}
//CHECK: {{^}}          ([[YELLOW]]TextComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:6[[RESET]], [[Yellow]]col:16[[RESET]]> Text=" A variable")))){{$}}
//CHECK: {{^}}    ([[GREEN]]FieldDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:24:3[[RESET]], [[Yellow]]col:7[[RESET]]>[[CYAN]] var2[[RESET]] [[Green]]'int'[[RESET]]{{$}}
//CHECK: {{^}}      ([[YELLOW]]FullComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:21:6[[RESET]], [[Yellow]]line:23:44[[RESET]]>{{$}}
//CHECK: {{^}}        ([[YELLOW]]ParagraphComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:21:6[[RESET]], [[Yellow]]col:22[[RESET]]>{{$}}
//CHECK: {{^}}          ([[YELLOW]]TextComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:6[[RESET]], [[Yellow]]col:22[[RESET]]> Text=" Another variable")){{$}}
//CHECK: {{^}}        ([[YELLOW]]ParagraphComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:23:6[[RESET]], [[Yellow]]col:44[[RESET]]>{{$}}
//CHECK: {{^}}          ([[YELLOW]]TextComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:6[[RESET]], [[Yellow]]col:44[[RESET]]> Text=" Like the other variable, but different")))){{$}}
//CHECK: {{^}}    ([[GREEN]]CXXConstructorDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:18:33[[RESET]]>[[CYAN]] Mutex[[RESET]] [[Green]]'void (void)'[[RESET]] inline{{$}}
//CHECK: {{^}}      ([[MAGENTA]]CompoundStmt[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:33[[RESET]]>)){{$}}
//CHECK: {{^}}    ([[GREEN]]CXXConstructorDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:33[[RESET]]>[[CYAN]] Mutex[[RESET]] [[Green]]'void (const class Mutex &)'[[RESET]] inline{{$}}
//CHECK: {{^}}      ([[GREEN]]ParmVarDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:33[[RESET]]> [[Green]]'const class Mutex &'[[RESET]])){{$}}
//CHECK: {{^}}    ([[GREEN]]CXXConstructorDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:33[[RESET]]>[[CYAN]] Mutex[[RESET]] [[Green]]'void (class Mutex &&)'[[RESET]] inline{{$}}
//CHECK: {{^}}      ([[GREEN]]ParmVarDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:33[[RESET]]> [[Green]]'class Mutex &&'[[RESET]]))){{$}}
//CHECK: {{^}}  ([[GREEN]]VarDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:1[[RESET]], [[Yellow]]line:25:3[[RESET]]>[[CYAN]] mu1[[RESET]] [[Green]]'class Mutex':'class Mutex'[[RESET]]{{$}}
//CHECK: {{^}}    ([[MAGENTA]]CXXConstructExpr[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:3[[RESET]]> [[Green]]'class Mutex':'class Mutex'[[RESET]][[Cyan]][[RESET]][[Cyan]][[RESET]] [[Green]]'void (void)'[[RESET]])){{$}}
//CHECK: {{^}}  ([[GREEN]]VarDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:18:1[[RESET]], [[Yellow]]line:25:8[[RESET]]>[[CYAN]] mu2[[RESET]] [[Green]]'class Mutex':'class Mutex'[[RESET]]{{$}}
//CHECK: {{^}}    ([[MAGENTA]]CXXConstructExpr[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:8[[RESET]]> [[Green]]'class Mutex':'class Mutex'[[RESET]][[Cyan]][[RESET]][[Cyan]][[RESET]] [[Green]]'void (void)'[[RESET]])){{$}}
//CHECK: {{^}}  ([[GREEN]]VarDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:26:1[[RESET]], [[Yellow]]col:5[[RESET]]>[[CYAN]] TestExpr[[RESET]] [[Green]]'int'[[RESET]]{{$}}
//CHECK: {{^}}    ([[BLUE]]GuardedByAttr[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:29[[RESET]]>{{$}}
//CHECK: {{^}}      ([[MAGENTA]]DeclRefExpr[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:40[[RESET]]> [[Green]]'class Mutex':'class Mutex'[[RESET]][[Cyan]] lvalue[[RESET]][[Cyan]][[RESET]] [[GREEN]]Var[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]][[CYAN]] 'mu1'[[RESET]] [[Green]]'class Mutex':'class Mutex'[[RESET]])))){{$}}

