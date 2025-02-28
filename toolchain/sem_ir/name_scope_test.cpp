// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/name_scope.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace Carbon::SemIR {
namespace {

using ::testing::ElementsAre;
using ::testing::Pair;

TEST(ScopeLookupResult, MakeWrappedLookupResultUsingExistingInstId) {
  InstId inst_id(1);
  auto result = ScopeLookupResult::MakeWrappedLookupResult(
      inst_id, AccessKind::Protected);

  EXPECT_FALSE(result.is_poisoned());
  EXPECT_TRUE(result.is_found());
  EXPECT_EQ(result.target_inst_id(), inst_id);
  EXPECT_DEATH(result.poisoning_loc_id(), "is_poisoned");
  EXPECT_EQ(result.access_kind(), AccessKind::Protected);
  EXPECT_TRUE(result == result);
}

TEST(ScopeLookupResult, MakeWrappedLookupResultUsingNoneInstId) {
  auto result = ScopeLookupResult::MakeWrappedLookupResult(
      InstId::None, AccessKind::Protected);

  EXPECT_FALSE(result.is_poisoned());
  EXPECT_FALSE(result.is_found());
  EXPECT_DEATH(result.target_inst_id(), "is_found");
  EXPECT_DEATH(result.poisoning_loc_id(), "is_poisoned");
  EXPECT_EQ(result.access_kind(), AccessKind::Protected);
  EXPECT_TRUE(result == result);
}

TEST(ScopeLookupResult, MakeWrappedLookupResultUsingErrorInst) {
  auto result = ScopeLookupResult::MakeWrappedLookupResult(
      ErrorInst::SingletonInstId, AccessKind::Private);

  EXPECT_FALSE(result.is_poisoned());
  EXPECT_TRUE(result.is_found());
  EXPECT_EQ(result.target_inst_id(), ErrorInst::SingletonInstId);
  EXPECT_DEATH(result.poisoning_loc_id(), "is_poisoned");
  EXPECT_EQ(result.access_kind(), AccessKind::Private);
  EXPECT_TRUE(result == result);
}

TEST(ScopeLookupResult, MakeFoundExisting) {
  InstId inst_id(1);
  auto result = ScopeLookupResult::MakeFound(inst_id, AccessKind::Protected);

  EXPECT_FALSE(result.is_poisoned());
  EXPECT_TRUE(result.is_found());
  EXPECT_EQ(result.target_inst_id(), inst_id);
  EXPECT_DEATH(result.poisoning_loc_id(), "is_poisoned");
  EXPECT_EQ(result.access_kind(), AccessKind::Protected);
  EXPECT_TRUE(result == result);
}

TEST(ScopeLookupResult, MakeFoundNone) {
  EXPECT_DEATH(
      ScopeLookupResult::MakeFound(InstId::None, AccessKind::Protected),
      "has_value");
}

TEST(ScopeLookupResult, MakeNotFound) {
  auto result = ScopeLookupResult::MakeNotFound();

  EXPECT_FALSE(result.is_poisoned());
  EXPECT_FALSE(result.is_found());
  EXPECT_DEATH(result.target_inst_id(), "is_found");
  EXPECT_DEATH(result.poisoning_loc_id(), "is_poisoned");
  EXPECT_EQ(result.access_kind(), AccessKind::Public);
  EXPECT_TRUE(result == result);
}

TEST(ScopeLookupResult, MakePoisoned) {
  LocId loc_id(1);
  auto result = ScopeLookupResult::MakePoisoned(loc_id);

  EXPECT_TRUE(result.is_poisoned());
  EXPECT_FALSE(result.is_found());
  EXPECT_DEATH(result.target_inst_id(), "is_found");
  EXPECT_EQ(result.poisoning_loc_id(), loc_id);
  EXPECT_EQ(result.access_kind(), AccessKind::Public);
  EXPECT_TRUE(result == result);
}

TEST(ScopeLookupResult, MakeError) {
  auto result = ScopeLookupResult::MakeError();

  EXPECT_FALSE(result.is_poisoned());
  EXPECT_TRUE(result.is_found());
  EXPECT_EQ(result.target_inst_id(), ErrorInst::SingletonInstId);
  EXPECT_DEATH(result.poisoning_loc_id(), "is_poisoned");
  EXPECT_EQ(result.access_kind(), AccessKind::Public);
  EXPECT_TRUE(result == result);
}

TEST(ScopeLookupResult, EqualityPoisonedDifferent) {
  EXPECT_FALSE(ScopeLookupResult::MakePoisoned(LocId(1)) ==
               ScopeLookupResult::MakeNotFound());
  EXPECT_FALSE(ScopeLookupResult::MakeNotFound() ==
               ScopeLookupResult::MakePoisoned(LocId(1)));
}

TEST(ScopeLookupResult, EqualityPoisonedLocIdDifferent) {
  EXPECT_FALSE(ScopeLookupResult::MakePoisoned(LocId(1)) ==
               ScopeLookupResult::MakePoisoned(LocId(2)));
}

TEST(ScopeLookupResult, EqualityFoundDifferent) {
  EXPECT_FALSE(ScopeLookupResult::MakeFound(InstId(1), AccessKind::Public) ==
               ScopeLookupResult::MakeNotFound());
  EXPECT_FALSE(ScopeLookupResult::MakeNotFound() ==
               ScopeLookupResult::MakeFound(InstId(1), AccessKind::Public));
}

TEST(ScopeLookupResult, EqualityFoundTargetInstIdDifferent) {
  EXPECT_FALSE(ScopeLookupResult::MakeFound(InstId(1), AccessKind::Public) ==
               ScopeLookupResult::MakeFound(InstId(2), AccessKind::Public));
}

TEST(ScopeLookupResult, EqualityFoundAccessKindDifferent) {
  EXPECT_FALSE(ScopeLookupResult::MakeFound(InstId(1), AccessKind::Public) ==
               ScopeLookupResult::MakeFound(InstId(1), AccessKind::Protected));
}

TEST(ScopeLookupResult, EqualityErrorDifferent) {
  EXPECT_FALSE(ScopeLookupResult::MakeNotFound() ==
               ScopeLookupResult::MakeError());
  EXPECT_FALSE(ScopeLookupResult::MakeError() ==
               ScopeLookupResult::MakeNotFound());
}

TEST(NameScope, Empty) {
  int id = 0;

  InstId scope_inst_id(++id);
  NameId scope_name_id(++id);
  NameScopeId parent_scope_id(++id);
  NameScope name_scope(scope_inst_id, scope_name_id, parent_scope_id);

  EXPECT_THAT(name_scope.entries(), ElementsAre());
  EXPECT_THAT(name_scope.extended_scopes(), ElementsAre());
  EXPECT_EQ(name_scope.inst_id(), scope_inst_id);
  EXPECT_EQ(name_scope.name_id(), scope_name_id);
  EXPECT_EQ(name_scope.parent_scope_id(), parent_scope_id);
  EXPECT_FALSE(name_scope.has_error());
  EXPECT_FALSE(name_scope.is_closed_import());
  EXPECT_FALSE(name_scope.is_imported_package());
  EXPECT_THAT(name_scope.import_ir_scopes(), ElementsAre());
}

TEST(NameScope, Lookup) {
  int id = 0;

  InstId scope_inst_id(++id);
  NameId scope_name_id(++id);
  NameScopeId parent_scope_id(++id);
  NameScope name_scope(scope_inst_id, scope_name_id, parent_scope_id);

  NameScope::Entry entry1 = {
      .name_id = NameId(++id),
      .result = ScopeLookupResult::MakeFound(InstId(++id), AccessKind::Public)};
  name_scope.AddRequired(entry1);

  NameScope::Entry entry2 = {.name_id = NameId(++id),
                             .result = ScopeLookupResult::MakeFound(
                                 InstId(++id), AccessKind::Protected)};
  name_scope.AddRequired(entry2);

  NameScope::Entry entry3 = {.name_id = NameId(++id),
                             .result = ScopeLookupResult::MakeFound(
                                 InstId(++id), AccessKind::Private)};
  name_scope.AddRequired(entry3);

  auto lookup = name_scope.Lookup(entry1.name_id);
  ASSERT_NE(lookup, std::nullopt);
  EXPECT_EQ(static_cast<NameScope&>(name_scope).GetEntry(*lookup), entry1);
  EXPECT_EQ(static_cast<const NameScope&>(name_scope).GetEntry(*lookup),
            entry1);

  lookup = name_scope.Lookup(entry2.name_id);
  ASSERT_NE(lookup, std::nullopt);
  EXPECT_EQ(name_scope.GetEntry(*lookup), entry2);

  lookup = name_scope.Lookup(entry3.name_id);
  ASSERT_NE(lookup, std::nullopt);
  EXPECT_EQ(name_scope.GetEntry(*lookup), entry3);

  NameId unknown_name_id(++id);
  EXPECT_EQ(name_scope.Lookup(unknown_name_id), std::nullopt);
  // Check that this is different from LookupOrPoison() - doesn't get poisoned.
  EXPECT_EQ(name_scope.Lookup(unknown_name_id), std::nullopt);
}

TEST(NameScope, LookupOrPoison) {
  int id = 0;

  InstId scope_inst_id(++id);
  NameId scope_name_id(++id);
  NameScopeId parent_scope_id(++id);
  NameScope name_scope(scope_inst_id, scope_name_id, parent_scope_id);

  NameScope::Entry entry1 = {
      .name_id = NameId(++id),
      .result = ScopeLookupResult::MakeFound(InstId(++id), AccessKind::Public)};
  name_scope.AddRequired(entry1);

  NameScope::Entry entry2 = {.name_id = NameId(++id),
                             .result = ScopeLookupResult::MakeFound(
                                 InstId(++id), AccessKind::Protected)};
  name_scope.AddRequired(entry2);

  NameScope::Entry entry3 = {.name_id = NameId(++id),
                             .result = ScopeLookupResult::MakeFound(
                                 InstId(++id), AccessKind::Private)};
  name_scope.AddRequired(entry3);

  LocId poisoning_loc_id_known_entries(++id);
  auto lookup =
      name_scope.LookupOrPoison(poisoning_loc_id_known_entries, entry1.name_id);
  ASSERT_NE(lookup, std::nullopt);
  EXPECT_EQ(static_cast<NameScope&>(name_scope).GetEntry(*lookup), entry1);
  EXPECT_EQ(static_cast<const NameScope&>(name_scope).GetEntry(*lookup),
            entry1);

  lookup =
      name_scope.LookupOrPoison(poisoning_loc_id_known_entries, entry2.name_id);
  ASSERT_NE(lookup, std::nullopt);
  EXPECT_EQ(name_scope.GetEntry(*lookup), entry2);

  lookup =
      name_scope.LookupOrPoison(poisoning_loc_id_known_entries, entry3.name_id);
  ASSERT_NE(lookup, std::nullopt);
  EXPECT_EQ(name_scope.GetEntry(*lookup), entry3);

  NameId unknown_name_id(++id);
  LocId poisoning_loc_id_unknown_entry(++id);
  EXPECT_EQ(name_scope.LookupOrPoison(poisoning_loc_id_unknown_entry,
                                      unknown_name_id),
            std::nullopt);
  // Check that this is different from Lookup() - does get poisoned.
  lookup = name_scope.Lookup(unknown_name_id);
  ASSERT_NE(lookup, std::nullopt);
  EXPECT_EQ(name_scope.GetEntry(*lookup).result,
            ScopeLookupResult::MakePoisoned(poisoning_loc_id_unknown_entry));
}

TEST(NameScope, LookupOrPoisonNotIdentifier) {
  int id = 0;

  InstId scope_inst_id(++id);
  NameId scope_name_id(++id);
  NameScopeId parent_scope_id(++id);
  NameScope name_scope(scope_inst_id, scope_name_id, parent_scope_id);
  LocId poisoning_loc_id(++id);

  EXPECT_EQ(name_scope.LookupOrPoison(poisoning_loc_id, NameId::SelfType),
            std::nullopt);
  // Check that this is different from the identifier use case - doesn't get
  // poisoned.
  EXPECT_EQ(name_scope.Lookup(NameId::SelfType), std::nullopt);
}

TEST(NameScope, LookupOrAdd) {
  int id = 0;

  InstId scope_inst_id(++id);
  NameId scope_name_id(++id);
  NameScopeId parent_scope_id(++id);
  NameScope name_scope(scope_inst_id, scope_name_id, parent_scope_id);

  NameScope::Entry entry1 = {
      .name_id = NameId(++id),
      .result = ScopeLookupResult::MakeFound(InstId(++id), AccessKind::Public)};
  {
    auto [added, entry_id] =
        name_scope.LookupOrAdd(entry1.name_id, entry1.result.target_inst_id(),
                               entry1.result.access_kind());
    EXPECT_TRUE(added);
    EXPECT_EQ(name_scope.GetEntry(entry_id), entry1);
  }

  NameScope::Entry entry2 = {.name_id = NameId(++id),
                             .result = ScopeLookupResult::MakeFound(
                                 InstId(++id), AccessKind::Protected)};
  {
    auto [added, entry_id] =
        name_scope.LookupOrAdd(entry2.name_id, entry2.result.target_inst_id(),
                               entry2.result.access_kind());
    EXPECT_TRUE(added);
    EXPECT_EQ(name_scope.GetEntry(entry_id), entry2);
  }

  NameScope::Entry entry3 = {.name_id = NameId(++id),
                             .result = ScopeLookupResult::MakeFound(
                                 InstId(++id), AccessKind::Private)};
  {
    auto [added, entry_id] =
        name_scope.LookupOrAdd(entry3.name_id, entry3.result.target_inst_id(),
                               entry3.result.access_kind());
    EXPECT_TRUE(added);
    EXPECT_EQ(name_scope.GetEntry(entry_id), entry3);
  }

  {
    auto [added, entry_id] =
        name_scope.LookupOrAdd(entry1.name_id, entry1.result.target_inst_id(),
                               entry1.result.access_kind());
    EXPECT_FALSE(added);
    EXPECT_EQ(name_scope.GetEntry(entry_id), entry1);
  }

  {
    auto [added, entry_id] =
        name_scope.LookupOrAdd(entry2.name_id, entry2.result.target_inst_id(),
                               entry2.result.access_kind());
    EXPECT_FALSE(added);
    EXPECT_EQ(name_scope.GetEntry(entry_id), entry2);
  }

  {
    auto [added, entry_id] =
        name_scope.LookupOrAdd(entry3.name_id, entry3.result.target_inst_id(),
                               entry3.result.access_kind());
    EXPECT_FALSE(added);
    EXPECT_EQ(name_scope.GetEntry(entry_id), entry3);
  }
}

TEST(NameScope, Poison) {
  int id = 0;

  InstId scope_inst_id(++id);
  NameId scope_name_id(++id);
  NameScopeId parent_scope_id(++id);
  NameScope name_scope(scope_inst_id, scope_name_id, parent_scope_id);

  NameId poison1(++id);
  LocId poisoning_loc1(++id);
  EXPECT_EQ(name_scope.LookupOrPoison(poisoning_loc1, poison1), std::nullopt);
  EXPECT_THAT(
      name_scope.entries(),
      ElementsAre(NameScope::Entry(
          {.name_id = poison1,
           .result = ScopeLookupResult::MakePoisoned(poisoning_loc1)})));

  NameId poison2(++id);
  LocId poisoning_loc2(++id);
  EXPECT_EQ(name_scope.LookupOrPoison(poisoning_loc2, poison2), std::nullopt);
  EXPECT_THAT(
      name_scope.entries(),
      ElementsAre(
          NameScope::Entry(
              {.name_id = poison1,
               .result = ScopeLookupResult::MakePoisoned(poisoning_loc1)}),
          NameScope::Entry(
              {.name_id = poison2,
               .result = ScopeLookupResult::MakePoisoned(poisoning_loc2)})));

  auto lookup = name_scope.Lookup(poison1);
  ASSERT_NE(lookup, std::nullopt);
  EXPECT_THAT(name_scope.GetEntry(*lookup),
              NameScope::Entry(
                  {.name_id = poison1,
                   .result = ScopeLookupResult::MakePoisoned(poisoning_loc1)}));
}

TEST(NameScope, AddRequiredAfterPoison) {
  int id = 0;

  InstId scope_inst_id(++id);
  NameId scope_name_id(++id);
  NameScopeId parent_scope_id(++id);
  NameScope name_scope(scope_inst_id, scope_name_id, parent_scope_id);

  NameId name_id(++id);
  InstId inst_id(++id);
  LocId poisoning_loc_id(++id);

  EXPECT_EQ(name_scope.LookupOrPoison(poisoning_loc_id, name_id), std::nullopt);
  EXPECT_THAT(
      name_scope.entries(),
      ElementsAre(NameScope::Entry(
          {.name_id = name_id,
           .result = ScopeLookupResult::MakePoisoned(poisoning_loc_id)})));

  NameScope::Entry entry = {
      .name_id = name_id,
      .result = ScopeLookupResult::MakeFound(inst_id, AccessKind::Private)};
  name_scope.AddRequired(entry);

  auto lookup = name_scope.LookupOrPoison(poisoning_loc_id, name_id);
  ASSERT_NE(lookup, std::nullopt);
  EXPECT_EQ(name_scope.GetEntry(*lookup),
            NameScope::Entry({.name_id = name_id,
                              .result = ScopeLookupResult::MakeFound(
                                  inst_id, AccessKind::Private)}));
}

TEST(NameScope, ExtendedScopes) {
  int id = 0;

  InstId scope_inst_id(++id);
  NameId scope_name_id(++id);
  NameScopeId parent_scope_id = NameScopeId::Package;
  NameScope name_scope(scope_inst_id, scope_name_id, parent_scope_id);

  EXPECT_THAT(name_scope.extended_scopes(), ElementsAre());

  InstId extended_scope1(++id);
  name_scope.AddExtendedScope(extended_scope1);
  EXPECT_THAT(name_scope.extended_scopes(), ElementsAre(extended_scope1));

  InstId extended_scope2(++id);
  name_scope.AddExtendedScope(extended_scope2);
  EXPECT_THAT(name_scope.extended_scopes(),
              ElementsAre(extended_scope1, extended_scope2));
}

TEST(NameScope, HasError) {
  int id = 0;

  InstId scope_inst_id(++id);
  NameId scope_name_id(++id);
  NameScopeId parent_scope_id(++id);
  NameScope name_scope(scope_inst_id, scope_name_id, parent_scope_id);

  EXPECT_FALSE(name_scope.has_error());

  name_scope.set_has_error();
  EXPECT_TRUE(name_scope.has_error());

  name_scope.set_has_error();
  EXPECT_TRUE(name_scope.has_error());
}

TEST(NameScope, IsClosedImport) {
  int id = 0;

  InstId scope_inst_id(++id);
  NameId scope_name_id(++id);
  NameScopeId parent_scope_id(++id);
  NameScope name_scope(scope_inst_id, scope_name_id, parent_scope_id);

  EXPECT_FALSE(name_scope.is_closed_import());

  name_scope.set_is_closed_import(true);
  EXPECT_TRUE(name_scope.is_closed_import());

  name_scope.set_is_closed_import(false);
  EXPECT_FALSE(name_scope.is_closed_import());
}

TEST(NameScope, IsImportedPackageParentNonPackageScope) {
  int id = 0;

  InstId scope_inst_id(++id);
  NameId scope_name_id(++id);
  NameScopeId parent_scope_id(++id);
  NameScope name_scope(scope_inst_id, scope_name_id, parent_scope_id);

  EXPECT_FALSE(name_scope.is_imported_package());

  name_scope.set_is_closed_import(true);
  EXPECT_FALSE(name_scope.is_imported_package());

  name_scope.set_is_closed_import(false);
  EXPECT_FALSE(name_scope.is_imported_package());
}

TEST(NameScope, IsImportedPackageParentPackageScope) {
  int id = 0;

  InstId scope_inst_id(++id);
  NameId scope_name_id(++id);
  NameScopeId parent_scope_id = NameScopeId::Package;
  NameScope name_scope(scope_inst_id, scope_name_id, parent_scope_id);

  EXPECT_FALSE(name_scope.is_imported_package());

  name_scope.set_is_closed_import(true);
  EXPECT_TRUE(name_scope.is_imported_package());

  name_scope.set_is_closed_import(false);
  EXPECT_FALSE(name_scope.is_imported_package());
}

TEST(NameScope, ImportIRScopes) {
  int id = 0;

  InstId scope_inst_id(++id);
  NameId scope_name_id(++id);
  NameScopeId parent_scope_id = NameScopeId::Package;
  NameScope name_scope(scope_inst_id, scope_name_id, parent_scope_id);

  EXPECT_THAT(name_scope.import_ir_scopes(), ElementsAre());

  ImportIRId import_ir_id1(++id);
  NameScopeId import_name_scope_id1(++id);
  name_scope.AddImportIRScope({import_ir_id1, import_name_scope_id1});
  EXPECT_THAT(name_scope.import_ir_scopes(),
              ElementsAre(Pair(import_ir_id1, import_name_scope_id1)));

  ImportIRId import_ir_id2(++id);
  NameScopeId import_name_scope_id2(++id);
  name_scope.AddImportIRScope({import_ir_id2, import_name_scope_id2});
  EXPECT_THAT(name_scope.import_ir_scopes(),
              ElementsAre(Pair(import_ir_id1, import_name_scope_id1),
                          Pair(import_ir_id2, import_name_scope_id2)));
}

}  // namespace
}  // namespace Carbon::SemIR
