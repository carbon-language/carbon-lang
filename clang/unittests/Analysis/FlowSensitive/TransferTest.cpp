//===- unittests/Analysis/FlowSensitive/TransferTest.cpp ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NoopAnalysis.h"
#include "TestingSupport.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cassert>
#include <string>
#include <utility>

namespace {

using namespace clang;
using namespace dataflow;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::IsNull;
using ::testing::NotNull;
using ::testing::Pair;

class TransferTest : public ::testing::Test {
protected:
  template <typename Matcher>
  void runDataflow(llvm::StringRef Code, Matcher Match) {
    test::checkDataflow<NoopAnalysis>(
        Code, "target",
        [](ASTContext &C, Environment &) { return NoopAnalysis(C); },
        [&Match](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) { Match(Results, ASTCtx); },
        {"-fsyntax-only", "-std=c++17"});
  }
};

/// Returns the `ValueDecl` for the given identifier.
///
/// Requirements:
///
///  `Name` must be unique in `ASTCtx`.
static const ValueDecl *findValueDecl(ASTContext &ASTCtx,
                                      llvm::StringRef Name) {
  auto TargetNodes = ast_matchers::match(
      ast_matchers::valueDecl(ast_matchers::hasName(Name)).bind("v"), ASTCtx);
  assert(TargetNodes.size() == 1 && "Name must be unique");
  auto *const Result = ast_matchers::selectFirst<ValueDecl>("v", TargetNodes);
  assert(Result != nullptr);
  return Result;
}

TEST_F(TransferTest, IntVarDecl) {
  std::string Code = R"(
    void target() {
      int Foo;
      // [[p]]
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const StorageLocation *FooLoc =
            Env.getStorageLocation(*FooDecl, SkipPast::None);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

        const Value *FooVal = Env.getValue(*FooLoc);
        EXPECT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));
      });
}

TEST_F(TransferTest, StructVarDecl) {
  std::string Code = R"(
    struct A {
      int Bar;
    };

    void target() {
      A Foo;
      // [[p]]
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        ASSERT_TRUE(FooDecl->getType()->isStructureType());
        auto FooFields = FooDecl->getType()->getAsRecordDecl()->fields();

        FieldDecl *BarDecl = nullptr;
        for (FieldDecl *Field : FooFields) {
          if (Field->getNameAsString() == "Bar") {
            BarDecl = Field;
          } else {
            FAIL() << "Unexpected field: " << Field->getNameAsString();
          }
        }
        ASSERT_THAT(BarDecl, NotNull());

        const auto *FooLoc = cast<AggregateStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *BarLoc =
            cast<ScalarStorageLocation>(&FooLoc->getChild(*BarDecl));

        const auto *FooVal = cast<StructValue>(Env.getValue(*FooLoc));
        const auto *BarVal = cast<IntegerValue>(&FooVal->getChild(*BarDecl));
        EXPECT_EQ(Env.getValue(*BarLoc), BarVal);
      });
}

TEST_F(TransferTest, ClassVarDecl) {
  std::string Code = R"(
    class A {
      int Bar;
    };

    void target() {
      A Foo;
      // [[p]]
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        ASSERT_TRUE(FooDecl->getType()->isClassType());
        auto FooFields = FooDecl->getType()->getAsRecordDecl()->fields();

        FieldDecl *BarDecl = nullptr;
        for (FieldDecl *Field : FooFields) {
          if (Field->getNameAsString() == "Bar") {
            BarDecl = Field;
          } else {
            FAIL() << "Unexpected field: " << Field->getNameAsString();
          }
        }
        ASSERT_THAT(BarDecl, NotNull());

        const auto *FooLoc = cast<AggregateStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *BarLoc =
            cast<ScalarStorageLocation>(&FooLoc->getChild(*BarDecl));

        const auto *FooVal = cast<StructValue>(Env.getValue(*FooLoc));
        const auto *BarVal = cast<IntegerValue>(&FooVal->getChild(*BarDecl));
        EXPECT_EQ(Env.getValue(*BarLoc), BarVal);
      });
}

TEST_F(TransferTest, ReferenceVarDecl) {
  std::string Code = R"(
    struct A {};

    A &getA();

    void target() {
      A &Foo = getA();
      // [[p]]
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const StorageLocation *FooLoc =
            Env.getStorageLocation(*FooDecl, SkipPast::None);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

        const ReferenceValue *FooVal =
            cast<ReferenceValue>(Env.getValue(*FooLoc));
        const StorageLocation &FooPointeeLoc = FooVal->getPointeeLoc();
        EXPECT_TRUE(isa<AggregateStorageLocation>(&FooPointeeLoc));

        const Value *FooPointeeVal = Env.getValue(FooPointeeLoc);
        EXPECT_TRUE(isa_and_nonnull<StructValue>(FooPointeeVal));
      });
}

TEST_F(TransferTest, SelfReferentialReferenceVarDecl) {
  std::string Code = R"(
    struct A;

    struct B {};

    struct C {
      A &FooRef;
      A *FooPtr;
      B &BazRef;
      B *BazPtr;
    };

    struct A {
      C &Bar;
    };

    A &getA();

    void target() {
      A &Foo = getA();
      // [[p]]
    }
  )";
  runDataflow(Code, [](llvm::ArrayRef<std::pair<
                           std::string, DataflowAnalysisState<NoopLattice>>>
                           Results,
                       ASTContext &ASTCtx) {
    ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
    const Environment &Env = Results[0].second.Env;

    const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
    ASSERT_THAT(FooDecl, NotNull());

    ASSERT_TRUE(FooDecl->getType()->isReferenceType());
    ASSERT_TRUE(FooDecl->getType().getNonReferenceType()->isStructureType());
    const auto FooFields =
        FooDecl->getType().getNonReferenceType()->getAsRecordDecl()->fields();

    FieldDecl *BarDecl = nullptr;
    for (FieldDecl *Field : FooFields) {
      if (Field->getNameAsString() == "Bar") {
        BarDecl = Field;
      } else {
        FAIL() << "Unexpected field: " << Field->getNameAsString();
      }
    }
    ASSERT_THAT(BarDecl, NotNull());

    ASSERT_TRUE(BarDecl->getType()->isReferenceType());
    ASSERT_TRUE(BarDecl->getType().getNonReferenceType()->isStructureType());
    const auto BarFields =
        BarDecl->getType().getNonReferenceType()->getAsRecordDecl()->fields();

    FieldDecl *FooRefDecl = nullptr;
    FieldDecl *FooPtrDecl = nullptr;
    FieldDecl *BazRefDecl = nullptr;
    FieldDecl *BazPtrDecl = nullptr;
    for (FieldDecl *Field : BarFields) {
      if (Field->getNameAsString() == "FooRef") {
        FooRefDecl = Field;
      } else if (Field->getNameAsString() == "FooPtr") {
        FooPtrDecl = Field;
      } else if (Field->getNameAsString() == "BazRef") {
        BazRefDecl = Field;
      } else if (Field->getNameAsString() == "BazPtr") {
        BazPtrDecl = Field;
      } else {
        FAIL() << "Unexpected field: " << Field->getNameAsString();
      }
    }
    ASSERT_THAT(FooRefDecl, NotNull());
    ASSERT_THAT(FooPtrDecl, NotNull());
    ASSERT_THAT(BazRefDecl, NotNull());
    ASSERT_THAT(BazPtrDecl, NotNull());

    const auto *FooLoc = cast<ScalarStorageLocation>(
        Env.getStorageLocation(*FooDecl, SkipPast::None));
    const auto *FooVal = cast<ReferenceValue>(Env.getValue(*FooLoc));
    const auto *FooPointeeVal =
        cast<StructValue>(Env.getValue(FooVal->getPointeeLoc()));

    const auto *BarVal =
        cast<ReferenceValue>(&FooPointeeVal->getChild(*BarDecl));
    const auto *BarPointeeVal =
        cast<StructValue>(Env.getValue(BarVal->getPointeeLoc()));

    const auto *FooRefVal =
        cast<ReferenceValue>(&BarPointeeVal->getChild(*FooRefDecl));
    const StorageLocation &FooRefPointeeLoc = FooRefVal->getPointeeLoc();
    EXPECT_THAT(Env.getValue(FooRefPointeeLoc), IsNull());

    const auto *FooPtrVal =
        cast<PointerValue>(&BarPointeeVal->getChild(*FooPtrDecl));
    const StorageLocation &FooPtrPointeeLoc = FooPtrVal->getPointeeLoc();
    EXPECT_THAT(Env.getValue(FooPtrPointeeLoc), IsNull());

    const auto *BazRefVal =
        cast<ReferenceValue>(&BarPointeeVal->getChild(*BazRefDecl));
    const StorageLocation &BazRefPointeeLoc = BazRefVal->getPointeeLoc();
    EXPECT_THAT(Env.getValue(BazRefPointeeLoc), NotNull());

    const auto *BazPtrVal =
        cast<PointerValue>(&BarPointeeVal->getChild(*BazPtrDecl));
    const StorageLocation &BazPtrPointeeLoc = BazPtrVal->getPointeeLoc();
    EXPECT_THAT(Env.getValue(BazPtrPointeeLoc), NotNull());
  });
}

TEST_F(TransferTest, PointerVarDecl) {
  std::string Code = R"(
    struct A {};

    A *getA();

    void target() {
      A *Foo = getA();
      // [[p]]
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const StorageLocation *FooLoc =
            Env.getStorageLocation(*FooDecl, SkipPast::None);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

        const PointerValue *FooVal = cast<PointerValue>(Env.getValue(*FooLoc));
        const StorageLocation &FooPointeeLoc = FooVal->getPointeeLoc();
        EXPECT_TRUE(isa<AggregateStorageLocation>(&FooPointeeLoc));

        const Value *FooPointeeVal = Env.getValue(FooPointeeLoc);
        EXPECT_TRUE(isa_and_nonnull<StructValue>(FooPointeeVal));
      });
}

TEST_F(TransferTest, SelfReferentialPointerVarDecl) {
  std::string Code = R"(
    struct A;

    struct B {};

    struct C {
      A &FooRef;
      A *FooPtr;
      B &BazRef;
      B *BazPtr;
    };

    struct A {
      C *Bar;
    };

    A *getA();

    void target() {
      A *Foo = getA();
      // [[p]]
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        ASSERT_TRUE(FooDecl->getType()->isPointerType());
        ASSERT_TRUE(FooDecl->getType()
                        ->getAs<PointerType>()
                        ->getPointeeType()
                        ->isStructureType());
        const auto FooFields = FooDecl->getType()
                                   ->getAs<PointerType>()
                                   ->getPointeeType()
                                   ->getAsRecordDecl()
                                   ->fields();

        FieldDecl *BarDecl = nullptr;
        for (FieldDecl *Field : FooFields) {
          if (Field->getNameAsString() == "Bar") {
            BarDecl = Field;
          } else {
            FAIL() << "Unexpected field: " << Field->getNameAsString();
          }
        }
        ASSERT_THAT(BarDecl, NotNull());

        ASSERT_TRUE(BarDecl->getType()->isPointerType());
        ASSERT_TRUE(BarDecl->getType()
                        ->getAs<PointerType>()
                        ->getPointeeType()
                        ->isStructureType());
        const auto BarFields = BarDecl->getType()
                                   ->getAs<PointerType>()
                                   ->getPointeeType()
                                   ->getAsRecordDecl()
                                   ->fields();

        FieldDecl *FooRefDecl = nullptr;
        FieldDecl *FooPtrDecl = nullptr;
        FieldDecl *BazRefDecl = nullptr;
        FieldDecl *BazPtrDecl = nullptr;
        for (FieldDecl *Field : BarFields) {
          if (Field->getNameAsString() == "FooRef") {
            FooRefDecl = Field;
          } else if (Field->getNameAsString() == "FooPtr") {
            FooPtrDecl = Field;
          } else if (Field->getNameAsString() == "BazRef") {
            BazRefDecl = Field;
          } else if (Field->getNameAsString() == "BazPtr") {
            BazPtrDecl = Field;
          } else {
            FAIL() << "Unexpected field: " << Field->getNameAsString();
          }
        }
        ASSERT_THAT(FooRefDecl, NotNull());
        ASSERT_THAT(FooPtrDecl, NotNull());
        ASSERT_THAT(BazRefDecl, NotNull());
        ASSERT_THAT(BazPtrDecl, NotNull());

        const auto *FooLoc = cast<ScalarStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *FooVal = cast<PointerValue>(Env.getValue(*FooLoc));
        const auto *FooPointeeVal =
            cast<StructValue>(Env.getValue(FooVal->getPointeeLoc()));

        const auto *BarVal =
            cast<PointerValue>(&FooPointeeVal->getChild(*BarDecl));
        const auto *BarPointeeVal =
            cast<StructValue>(Env.getValue(BarVal->getPointeeLoc()));

        const auto *FooRefVal =
            cast<ReferenceValue>(&BarPointeeVal->getChild(*FooRefDecl));
        const StorageLocation &FooRefPointeeLoc = FooRefVal->getPointeeLoc();
        EXPECT_THAT(Env.getValue(FooRefPointeeLoc), IsNull());

        const auto *FooPtrVal =
            cast<PointerValue>(&BarPointeeVal->getChild(*FooPtrDecl));
        const StorageLocation &FooPtrPointeeLoc = FooPtrVal->getPointeeLoc();
        EXPECT_THAT(Env.getValue(FooPtrPointeeLoc), IsNull());

        const auto *BazRefVal =
            cast<ReferenceValue>(&BarPointeeVal->getChild(*BazRefDecl));
        const StorageLocation &BazRefPointeeLoc = BazRefVal->getPointeeLoc();
        EXPECT_THAT(Env.getValue(BazRefPointeeLoc), NotNull());

        const auto *BazPtrVal =
            cast<PointerValue>(&BarPointeeVal->getChild(*BazPtrDecl));
        const StorageLocation &BazPtrPointeeLoc = BazPtrVal->getPointeeLoc();
        EXPECT_THAT(Env.getValue(BazPtrPointeeLoc), NotNull());
      });
}

TEST_F(TransferTest, JoinVarDecl) {
  std::string Code = R"(
    void target(bool B) {
      int Foo;
      // [[p1]]
      if (B) {
        int Bar;
        // [[p2]]
      } else {
        int Baz;
        // [[p3]]
      }
      (void)0;
      // [[p4]]
    }
  )";
  runDataflow(Code, [](llvm::ArrayRef<std::pair<
                           std::string, DataflowAnalysisState<NoopLattice>>>
                           Results,
                       ASTContext &ASTCtx) {
    ASSERT_THAT(Results, ElementsAre(Pair("p4", _), Pair("p3", _),
                                     Pair("p2", _), Pair("p1", _)));
    const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
    ASSERT_THAT(FooDecl, NotNull());

    const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
    ASSERT_THAT(BarDecl, NotNull());

    const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
    ASSERT_THAT(BazDecl, NotNull());

    const Environment &Env1 = Results[3].second.Env;
    const StorageLocation *FooLoc =
        Env1.getStorageLocation(*FooDecl, SkipPast::None);
    EXPECT_THAT(FooLoc, NotNull());
    EXPECT_THAT(Env1.getStorageLocation(*BarDecl, SkipPast::None), IsNull());
    EXPECT_THAT(Env1.getStorageLocation(*BazDecl, SkipPast::None), IsNull());

    const Environment &Env2 = Results[2].second.Env;
    EXPECT_EQ(Env2.getStorageLocation(*FooDecl, SkipPast::None), FooLoc);
    EXPECT_THAT(Env2.getStorageLocation(*BarDecl, SkipPast::None), NotNull());
    EXPECT_THAT(Env2.getStorageLocation(*BazDecl, SkipPast::None), IsNull());

    const Environment &Env3 = Results[1].second.Env;
    EXPECT_EQ(Env3.getStorageLocation(*FooDecl, SkipPast::None), FooLoc);
    EXPECT_THAT(Env3.getStorageLocation(*BarDecl, SkipPast::None), IsNull());
    EXPECT_THAT(Env3.getStorageLocation(*BazDecl, SkipPast::None), NotNull());

    const Environment &Env4 = Results[0].second.Env;
    EXPECT_EQ(Env4.getStorageLocation(*FooDecl, SkipPast::None), FooLoc);
    EXPECT_THAT(Env4.getStorageLocation(*BarDecl, SkipPast::None), IsNull());
    EXPECT_THAT(Env4.getStorageLocation(*BazDecl, SkipPast::None), IsNull());
  });
}

TEST_F(TransferTest, BinaryOperatorAssign) {
  std::string Code = R"(
    void target() {
      int Foo;
      int Bar;
      (Bar) = (Foo);
      // [[p]]
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const Value *FooVal = Env.getValue(*FooDecl, SkipPast::None);
                ASSERT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                EXPECT_EQ(Env.getValue(*BarDecl, SkipPast::None), FooVal);
              });
}

TEST_F(TransferTest, VarDeclInitAssign) {
  std::string Code = R"(
    void target() {
      int Foo;
      int Bar = Foo;
      // [[p]]
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const Value *FooVal = Env.getValue(*FooDecl, SkipPast::None);
                ASSERT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                EXPECT_EQ(Env.getValue(*BarDecl, SkipPast::None), FooVal);
              });
}

TEST_F(TransferTest, VarDeclInitAssignChained) {
  std::string Code = R"(
    void target() {
      int Foo;
      int Bar;
      int Baz = (Bar = Foo);
      // [[p]]
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const Value *FooVal = Env.getValue(*FooDecl, SkipPast::None);
                ASSERT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
                ASSERT_THAT(BazDecl, NotNull());

                EXPECT_EQ(Env.getValue(*BarDecl, SkipPast::None), FooVal);
                EXPECT_EQ(Env.getValue(*BazDecl, SkipPast::None), FooVal);
              });
}

TEST_F(TransferTest, VarDeclInitAssignPtrDeref) {
  std::string Code = R"(
    void target() {
      int Foo;
      int *Bar;
      *(Bar) = Foo;
      int Baz = *(Bar);
      // [[p]]
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const Value *FooVal = Env.getValue(*FooDecl, SkipPast::None);
                ASSERT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                const auto *BarVal =
                    cast<PointerValue>(Env.getValue(*BarDecl, SkipPast::None));
                EXPECT_EQ(Env.getValue(BarVal->getPointeeLoc()), FooVal);

                const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
                ASSERT_THAT(BazDecl, NotNull());

                EXPECT_EQ(Env.getValue(*BazDecl, SkipPast::None), FooVal);
              });
}

TEST_F(TransferTest, AssignToAndFromReference) {
  std::string Code = R"(
    void target() {
      int Foo;
      int Bar;
      int &Baz = Foo;
      // [[p1]]
      Baz = Bar;
      int Qux = Baz;
      int &Quux = Baz;
      // [[p2]]
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p1", _), Pair("p2", _)));
        const Environment &Env1 = Results[0].second.Env;
        const Environment &Env2 = Results[1].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const Value *FooVal = Env1.getValue(*FooDecl, SkipPast::None);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const Value *BarVal = Env1.getValue(*BarDecl, SkipPast::None);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(BarVal));

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        EXPECT_EQ(Env1.getValue(*BazDecl, SkipPast::Reference), FooVal);

        EXPECT_EQ(Env2.getValue(*BazDecl, SkipPast::Reference), BarVal);
        EXPECT_EQ(Env2.getValue(*FooDecl, SkipPast::None), BarVal);

        const ValueDecl *QuxDecl = findValueDecl(ASTCtx, "Qux");
        ASSERT_THAT(QuxDecl, NotNull());
        EXPECT_EQ(Env2.getValue(*QuxDecl, SkipPast::None), BarVal);

        const ValueDecl *QuuxDecl = findValueDecl(ASTCtx, "Quux");
        ASSERT_THAT(QuuxDecl, NotNull());
        EXPECT_EQ(Env2.getValue(*QuuxDecl, SkipPast::Reference), BarVal);
      });
}

TEST_F(TransferTest, MultipleParamDecls) {
  std::string Code = R"(
    void target(int Foo, int Bar) {
      (void)0;
      // [[p]]
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const StorageLocation *FooLoc =
                    Env.getStorageLocation(*FooDecl, SkipPast::None);
                ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

                const Value *FooVal = Env.getValue(*FooLoc);
                ASSERT_TRUE(isa_and_nonnull<IntegerValue>(FooVal));

                const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
                ASSERT_THAT(BarDecl, NotNull());

                const StorageLocation *BarLoc =
                    Env.getStorageLocation(*BarDecl, SkipPast::None);
                ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(BarLoc));

                const Value *BarVal = Env.getValue(*BarLoc);
                EXPECT_TRUE(isa_and_nonnull<IntegerValue>(BarVal));
              });
}

TEST_F(TransferTest, StructParamDecl) {
  std::string Code = R"(
    struct A {
      int Bar;
    };

    void target(A Foo) {
      (void)0;
      // [[p]]
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        ASSERT_TRUE(FooDecl->getType()->isStructureType());
        auto FooFields = FooDecl->getType()->getAsRecordDecl()->fields();

        FieldDecl *BarDecl = nullptr;
        for (FieldDecl *Field : FooFields) {
          if (Field->getNameAsString() == "Bar") {
            BarDecl = Field;
          } else {
            FAIL() << "Unexpected field: " << Field->getNameAsString();
          }
        }
        ASSERT_THAT(BarDecl, NotNull());

        const auto *FooLoc = cast<AggregateStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *BarLoc =
            cast<ScalarStorageLocation>(&FooLoc->getChild(*BarDecl));

        const auto *FooVal = cast<StructValue>(Env.getValue(*FooLoc));
        const auto *BarVal = cast<IntegerValue>(&FooVal->getChild(*BarDecl));
        EXPECT_EQ(Env.getValue(*BarLoc), BarVal);
      });
}

TEST_F(TransferTest, ReferenceParamDecl) {
  std::string Code = R"(
    struct A {};

    void target(A &Foo) {
      (void)0;
      // [[p]]
    }
  )";
  runDataflow(Code,
              [](llvm::ArrayRef<
                     std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                     Results,
                 ASTContext &ASTCtx) {
                ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
                const Environment &Env = Results[0].second.Env;

                const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
                ASSERT_THAT(FooDecl, NotNull());

                const StorageLocation *FooLoc =
                    Env.getStorageLocation(*FooDecl, SkipPast::None);
                ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

                const ReferenceValue *FooVal =
                    dyn_cast<ReferenceValue>(Env.getValue(*FooLoc));
                ASSERT_THAT(FooVal, NotNull());

                const StorageLocation &FooPointeeLoc = FooVal->getPointeeLoc();
                EXPECT_TRUE(isa<AggregateStorageLocation>(&FooPointeeLoc));

                const Value *FooPointeeVal = Env.getValue(FooPointeeLoc);
                EXPECT_TRUE(isa_and_nonnull<StructValue>(FooPointeeVal));
              });
}

TEST_F(TransferTest, PointerParamDecl) {
  std::string Code = R"(
    struct A {};

    void target(A *Foo) {
      (void)0;
      // [[p]]
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        const StorageLocation *FooLoc =
            Env.getStorageLocation(*FooDecl, SkipPast::None);
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(FooLoc));

        const PointerValue *FooVal = cast<PointerValue>(Env.getValue(*FooLoc));
        const StorageLocation &FooPointeeLoc = FooVal->getPointeeLoc();
        EXPECT_TRUE(isa<AggregateStorageLocation>(&FooPointeeLoc));

        const Value *FooPointeeVal = Env.getValue(FooPointeeLoc);
        EXPECT_TRUE(isa_and_nonnull<StructValue>(FooPointeeVal));
      });
}

TEST_F(TransferTest, StructMember) {
  std::string Code = R"(
    struct A {
      int Bar;
    };

    void target(A Foo) {
      int Baz = Foo.Bar;
      // [[p]]
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        ASSERT_TRUE(FooDecl->getType()->isStructureType());
        auto FooFields = FooDecl->getType()->getAsRecordDecl()->fields();

        FieldDecl *BarDecl = nullptr;
        for (FieldDecl *Field : FooFields) {
          if (Field->getNameAsString() == "Bar") {
            BarDecl = Field;
          } else {
            FAIL() << "Unexpected field: " << Field->getNameAsString();
          }
        }
        ASSERT_THAT(BarDecl, NotNull());

        const auto *FooLoc = cast<AggregateStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *FooVal = cast<StructValue>(Env.getValue(*FooLoc));
        const auto *BarVal = cast<IntegerValue>(&FooVal->getChild(*BarDecl));

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        EXPECT_EQ(Env.getValue(*BazDecl, SkipPast::None), BarVal);
      });
}

TEST_F(TransferTest, ClassMember) {
  std::string Code = R"(
    class A {
    public:
      int Bar;
    };

    void target(A Foo) {
      int Baz = Foo.Bar;
      // [[p]]
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        ASSERT_TRUE(FooDecl->getType()->isClassType());
        auto FooFields = FooDecl->getType()->getAsRecordDecl()->fields();

        FieldDecl *BarDecl = nullptr;
        for (FieldDecl *Field : FooFields) {
          if (Field->getNameAsString() == "Bar") {
            BarDecl = Field;
          } else {
            FAIL() << "Unexpected field: " << Field->getNameAsString();
          }
        }
        ASSERT_THAT(BarDecl, NotNull());

        const auto *FooLoc = cast<AggregateStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *FooVal = cast<StructValue>(Env.getValue(*FooLoc));
        const auto *BarVal = cast<IntegerValue>(&FooVal->getChild(*BarDecl));

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        EXPECT_EQ(Env.getValue(*BazDecl, SkipPast::None), BarVal);
      });
}

TEST_F(TransferTest, ReferenceMember) {
  std::string Code = R"(
    struct A {
      int &Bar;
    };

    void target(A Foo) {
      int Baz = Foo.Bar;
      // [[p]]
    }
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());

        ASSERT_TRUE(FooDecl->getType()->isStructureType());
        auto FooFields = FooDecl->getType()->getAsRecordDecl()->fields();

        FieldDecl *BarDecl = nullptr;
        for (FieldDecl *Field : FooFields) {
          if (Field->getNameAsString() == "Bar") {
            BarDecl = Field;
          } else {
            FAIL() << "Unexpected field: " << Field->getNameAsString();
          }
        }
        ASSERT_THAT(BarDecl, NotNull());

        const auto *FooLoc = cast<AggregateStorageLocation>(
            Env.getStorageLocation(*FooDecl, SkipPast::None));
        const auto *FooVal = cast<StructValue>(Env.getValue(*FooLoc));
        const auto *BarVal = cast<ReferenceValue>(&FooVal->getChild(*BarDecl));
        const auto *BarPointeeVal =
            cast<IntegerValue>(Env.getValue(BarVal->getPointeeLoc()));

        const ValueDecl *BazDecl = findValueDecl(ASTCtx, "Baz");
        ASSERT_THAT(BazDecl, NotNull());

        EXPECT_EQ(Env.getValue(*BazDecl, SkipPast::None), BarPointeeVal);
      });
}

TEST_F(TransferTest, StructThisMember) {
  std::string Code = R"(
    struct A {
      int Bar;

      struct B {
        int Baz;
      };

      B Qux;

      void target() {
        int Foo = Bar;
        int Quux = Qux.Baz;
        // [[p]]
      }
    };
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const auto *ThisLoc = dyn_cast<AggregateStorageLocation>(
            Env.getThisPointeeStorageLocation());
        ASSERT_THAT(ThisLoc, NotNull());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *BarLoc =
            cast<ScalarStorageLocation>(&ThisLoc->getChild(*BarDecl));
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(BarLoc));

        const Value *BarVal = Env.getValue(*BarLoc);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(BarVal));

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());
        EXPECT_EQ(Env.getValue(*FooDecl, SkipPast::None), BarVal);

        const ValueDecl *QuxDecl = findValueDecl(ASTCtx, "Qux");
        ASSERT_THAT(QuxDecl, NotNull());

        ASSERT_TRUE(QuxDecl->getType()->isStructureType());
        auto QuxFields = QuxDecl->getType()->getAsRecordDecl()->fields();

        FieldDecl *BazDecl = nullptr;
        for (FieldDecl *Field : QuxFields) {
          if (Field->getNameAsString() == "Baz") {
            BazDecl = Field;
          } else {
            FAIL() << "Unexpected field: " << Field->getNameAsString();
          }
        }
        ASSERT_THAT(BazDecl, NotNull());

        const auto *QuxLoc =
            cast<AggregateStorageLocation>(&ThisLoc->getChild(*QuxDecl));
        const auto *QuxVal = dyn_cast<StructValue>(Env.getValue(*QuxLoc));
        ASSERT_THAT(QuxVal, NotNull());

        const auto *BazLoc =
            cast<ScalarStorageLocation>(&QuxLoc->getChild(*BazDecl));
        const auto *BazVal = cast<IntegerValue>(&QuxVal->getChild(*BazDecl));
        EXPECT_EQ(Env.getValue(*BazLoc), BazVal);

        const ValueDecl *QuuxDecl = findValueDecl(ASTCtx, "Quux");
        ASSERT_THAT(QuuxDecl, NotNull());
        EXPECT_EQ(Env.getValue(*QuuxDecl, SkipPast::None), BazVal);
      });
}

TEST_F(TransferTest, ClassThisMember) {
  std::string Code = R"(
    class A {
      int Bar;

      class B {
      public:
        int Baz;
      };

      B Qux;

      void target() {
        int Foo = Bar;
        int Quux = Qux.Baz;
        // [[p]]
      }
    };
  )";
  runDataflow(
      Code, [](llvm::ArrayRef<
                   std::pair<std::string, DataflowAnalysisState<NoopLattice>>>
                   Results,
               ASTContext &ASTCtx) {
        ASSERT_THAT(Results, ElementsAre(Pair("p", _)));
        const Environment &Env = Results[0].second.Env;

        const auto *ThisLoc =
            cast<AggregateStorageLocation>(Env.getThisPointeeStorageLocation());

        const ValueDecl *BarDecl = findValueDecl(ASTCtx, "Bar");
        ASSERT_THAT(BarDecl, NotNull());

        const auto *BarLoc =
            cast<ScalarStorageLocation>(&ThisLoc->getChild(*BarDecl));
        ASSERT_TRUE(isa_and_nonnull<ScalarStorageLocation>(BarLoc));

        const Value *BarVal = Env.getValue(*BarLoc);
        ASSERT_TRUE(isa_and_nonnull<IntegerValue>(BarVal));

        const ValueDecl *FooDecl = findValueDecl(ASTCtx, "Foo");
        ASSERT_THAT(FooDecl, NotNull());
        EXPECT_EQ(Env.getValue(*FooDecl, SkipPast::None), BarVal);

        const ValueDecl *QuxDecl = findValueDecl(ASTCtx, "Qux");
        ASSERT_THAT(QuxDecl, NotNull());

        ASSERT_TRUE(QuxDecl->getType()->isClassType());
        auto QuxFields = QuxDecl->getType()->getAsRecordDecl()->fields();

        FieldDecl *BazDecl = nullptr;
        for (FieldDecl *Field : QuxFields) {
          if (Field->getNameAsString() == "Baz") {
            BazDecl = Field;
          } else {
            FAIL() << "Unexpected field: " << Field->getNameAsString();
          }
        }
        ASSERT_THAT(BazDecl, NotNull());

        const auto *QuxLoc =
            cast<AggregateStorageLocation>(&ThisLoc->getChild(*QuxDecl));
        const auto *QuxVal = dyn_cast<StructValue>(Env.getValue(*QuxLoc));
        ASSERT_THAT(QuxVal, NotNull());

        const auto *BazLoc =
            cast<ScalarStorageLocation>(&QuxLoc->getChild(*BazDecl));
        const auto *BazVal = cast<IntegerValue>(&QuxVal->getChild(*BazDecl));
        EXPECT_EQ(Env.getValue(*BazLoc), BazVal);

        const ValueDecl *QuuxDecl = findValueDecl(ASTCtx, "Quux");
        ASSERT_THAT(QuuxDecl, NotNull());
        EXPECT_EQ(Env.getValue(*QuuxDecl, SkipPast::None), BazVal);
      });
}

} // namespace
