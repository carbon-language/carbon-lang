//===- lld/unittest/WinLinkDriverTest.cpp ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Windows link.exe driver tests.
///
//===----------------------------------------------------------------------===//

#include "DriverTest.h"
#include "lld/ReaderWriter/PECOFFLinkingContext.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/COFF.h"
#include <set>
#include <vector>

using namespace llvm;
using namespace lld;

namespace {
class WinLinkParserTest
    : public ParserTest<WinLinkDriver, PECOFFLinkingContext> {
protected:
  const LinkingContext *linkingContext() override { return &_ctx; }
};
}

TEST_F(WinLinkParserTest, Basic) {
  EXPECT_TRUE(parse("link.exe", "/subsystem:console", "/out:a.exe",
                    "-entry:start", "a.obj", "b.obj", "c.obj", nullptr));
  EXPECT_EQ(llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_CUI, _ctx.getSubsystem());
  EXPECT_EQ(llvm::COFF::IMAGE_FILE_MACHINE_I386, _ctx.getMachineType());
  EXPECT_EQ("a.exe", _ctx.outputPath());
  EXPECT_EQ("start", _ctx.getEntrySymbolName());
  EXPECT_EQ(4, inputFileCount());
  EXPECT_EQ("a.obj", inputFile(0));
  EXPECT_EQ("b.obj", inputFile(1));
  EXPECT_EQ("c.obj", inputFile(2));
  EXPECT_TRUE(_ctx.getInputSearchPaths().empty());

  // Unspecified flags will have default values.
  EXPECT_FALSE(_ctx.isDll());
  EXPECT_EQ(6, _ctx.getMinOSVersion().majorVersion);
  EXPECT_EQ(0, _ctx.getMinOSVersion().minorVersion);
  EXPECT_EQ(0x400000U, _ctx.getBaseAddress());
  EXPECT_EQ(1024 * 1024U, _ctx.getStackReserve());
  EXPECT_EQ(4096U, _ctx.getStackCommit());
  EXPECT_EQ(4096U, _ctx.getSectionDefaultAlignment());
  EXPECT_FALSE(_ctx.allowRemainingUndefines());
  EXPECT_TRUE(_ctx.isNxCompat());
  EXPECT_FALSE(_ctx.getLargeAddressAware());
  EXPECT_TRUE(_ctx.getAllowBind());
  EXPECT_TRUE(_ctx.getAllowIsolation());
  EXPECT_FALSE(_ctx.getSwapRunFromCD());
  EXPECT_FALSE(_ctx.getSwapRunFromNet());
  EXPECT_TRUE(_ctx.getBaseRelocationEnabled());
  EXPECT_TRUE(_ctx.isTerminalServerAware());
  EXPECT_TRUE(_ctx.getDynamicBaseEnabled());
  EXPECT_TRUE(_ctx.getCreateManifest());
  EXPECT_EQ("", _ctx.getManifestDependency());
  EXPECT_FALSE(_ctx.getEmbedManifest());
  EXPECT_EQ(1, _ctx.getManifestId());
  EXPECT_TRUE(_ctx.getManifestUAC());
  EXPECT_EQ("'asInvoker'", _ctx.getManifestLevel());
  EXPECT_EQ("'false'", _ctx.getManifestUiAccess());
  EXPECT_TRUE(_ctx.deadStrip());
  EXPECT_FALSE(_ctx.logInputFiles());
}

TEST_F(WinLinkParserTest, StartsWithHyphen) {
  EXPECT_TRUE(
      parse("link.exe", "-subsystem:console", "-out:a.exe", "a.obj", nullptr));
  EXPECT_EQ(llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_CUI, _ctx.getSubsystem());
  EXPECT_EQ("a.exe", _ctx.outputPath());
  EXPECT_EQ(2, inputFileCount());
  EXPECT_EQ("a.obj", inputFile(0));
}

TEST_F(WinLinkParserTest, UppercaseOption) {
  EXPECT_TRUE(
      parse("link.exe", "/SUBSYSTEM:CONSOLE", "/OUT:a.exe", "a.obj", nullptr));
  EXPECT_EQ(llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_CUI, _ctx.getSubsystem());
  EXPECT_EQ("a.exe", _ctx.outputPath());
  EXPECT_EQ(2, inputFileCount());
  EXPECT_EQ("a.obj", inputFile(0));
}

TEST_F(WinLinkParserTest, Mllvm) {
  EXPECT_TRUE(parse("link.exe", "/mllvm:-debug", "a.obj", nullptr));
  const std::vector<const char *> &options = _ctx.llvmOptions();
  EXPECT_EQ(1U, options.size());
  EXPECT_STREQ("-debug", options[0]);
}

TEST_F(WinLinkParserTest, NoInputFiles) {
  EXPECT_FALSE(parse("link.exe", nullptr));
  EXPECT_EQ("No input files\n", errorMessage());
}

//
// Tests for implicit file extension interpolation.
//

TEST_F(WinLinkParserTest, NoFileExtension) {
  EXPECT_TRUE(parse("link.exe", "foo", "bar", nullptr));
  EXPECT_EQ("foo.exe", _ctx.outputPath());
  EXPECT_EQ(3, inputFileCount());
  EXPECT_EQ("foo.obj", inputFile(0));
  EXPECT_EQ("bar.obj", inputFile(1));
}

TEST_F(WinLinkParserTest, NonStandardFileExtension) {
  EXPECT_TRUE(parse("link.exe", "foo.o", nullptr));
  EXPECT_EQ("foo.exe", _ctx.outputPath());
  EXPECT_EQ(2, inputFileCount());
  EXPECT_EQ("foo.o", inputFile(0));
}

TEST_F(WinLinkParserTest, Libpath) {
  EXPECT_TRUE(
      parse("link.exe", "/libpath:dir1", "/libpath:dir2", "a.obj", nullptr));
  const std::vector<StringRef> &paths = _ctx.getInputSearchPaths();
  EXPECT_EQ(2U, paths.size());
  EXPECT_EQ("dir1", paths[0]);
  EXPECT_EQ("dir2", paths[1]);
}

//
// Tests for input file order
//

TEST_F(WinLinkParserTest, InputOrder) {
  EXPECT_TRUE(parse("link.exe", "a.lib", "b.obj", "c.obj", "a.lib", "d.obj",
                    nullptr));
  EXPECT_EQ(5, inputFileCount());
  EXPECT_EQ("b.obj", inputFile(0));
  EXPECT_EQ("c.obj", inputFile(1));
  EXPECT_EQ("d.obj", inputFile(2));
  EXPECT_EQ("a.lib", inputFile(3));
}

//
// Tests for command line options that take values.
//

TEST_F(WinLinkParserTest, AlternateName) {
  EXPECT_TRUE(parse("link.exe", "/alternatename:sym1=sym2", "a.out", nullptr));
  EXPECT_EQ("sym1", _ctx.getAlternateName("sym2"));
  EXPECT_EQ("", _ctx.getAlternateName("foo"));
}

TEST_F(WinLinkParserTest, Export) {
  EXPECT_TRUE(parse("link.exe", "/export:foo", "a.out", nullptr));
  const std::vector<PECOFFLinkingContext::ExportDesc> &exports =
      _ctx.getDllExports();
  EXPECT_EQ(1U, exports.size());
  EXPECT_EQ("_foo", exports[0].name);
  EXPECT_EQ(-1, exports[0].ordinal);
  EXPECT_FALSE(exports[0].noname);
  EXPECT_FALSE(exports[0].isData);
}

TEST_F(WinLinkParserTest, ExportWithOptions) {
  EXPECT_TRUE(parse("link.exe", "/export:foo,@8,noname,data",
                    "/export:bar,@10,data", "a.out", nullptr));
  const std::vector<PECOFFLinkingContext::ExportDesc> &exports =
      _ctx.getDllExports();
  EXPECT_EQ(2U, exports.size());
  EXPECT_EQ("_foo", exports[0].name);
  EXPECT_EQ(8, exports[0].ordinal);
  EXPECT_TRUE(exports[0].noname);
  EXPECT_TRUE(exports[0].isData);
  EXPECT_EQ("_bar", exports[1].name);
  EXPECT_EQ(10, exports[1].ordinal);
  EXPECT_FALSE(exports[1].noname);
  EXPECT_TRUE(exports[1].isData);
}

TEST_F(WinLinkParserTest, ExportDuplicateExports) {
  EXPECT_TRUE(
      parse("link.exe", "/export:foo", "/export:foo,@2", "a.out", nullptr));
  const std::vector<PECOFFLinkingContext::ExportDesc> &exports =
      _ctx.getDllExports();
  EXPECT_EQ(1U, exports.size());
  EXPECT_EQ("_foo", exports[0].name);
  EXPECT_EQ(-1, exports[0].ordinal);
}

TEST_F(WinLinkParserTest, ExportDuplicateOrdinals) {
  EXPECT_FALSE(
      parse("link.exe", "/export:foo,@1", "/export:bar,@1", "a.out", nullptr));
}

TEST_F(WinLinkParserTest, ExportInvalid1) {
  EXPECT_FALSE(parse("link.exe", "/export:foo,@0", "a.out", nullptr));
}

TEST_F(WinLinkParserTest, ExportInvalid2) {
  EXPECT_FALSE(parse("link.exe", "/export:foo,@65536", "a.out", nullptr));
}

TEST_F(WinLinkParserTest, MachineX86) {
  EXPECT_TRUE(parse("link.exe", "/machine:x86", "a.obj", nullptr));
  EXPECT_EQ(llvm::COFF::IMAGE_FILE_MACHINE_I386, _ctx.getMachineType());
}

TEST_F(WinLinkParserTest, MachineX64) {
  EXPECT_TRUE(parse("link.exe", "/machine:x64", "a.obj", nullptr));
  EXPECT_EQ(llvm::COFF::IMAGE_FILE_MACHINE_AMD64, _ctx.getMachineType());
}

TEST_F(WinLinkParserTest, MachineArm) {
  EXPECT_TRUE(parse("link.exe", "/machine:arm", "a.obj", nullptr));
  EXPECT_EQ(llvm::COFF::IMAGE_FILE_MACHINE_ARMNT, _ctx.getMachineType());
}

TEST_F(WinLinkParserTest, MachineUnknown) {
  EXPECT_FALSE(parse("link.exe", "/machine:nosucharch", "a.obj", nullptr));
  EXPECT_EQ("error: unknown machine type: nosucharch\n", errorMessage());
}

TEST_F(WinLinkParserTest, MajorImageVersion) {
  EXPECT_TRUE(parse("link.exe", "/version:7", "foo.o", nullptr));
  EXPECT_EQ(7, _ctx.getImageVersion().majorVersion);
  EXPECT_EQ(0, _ctx.getImageVersion().minorVersion);
}

TEST_F(WinLinkParserTest, MajorMinorImageVersion) {
  EXPECT_TRUE(parse("link.exe", "/version:72.35", "foo.o", nullptr));
  EXPECT_EQ(72, _ctx.getImageVersion().majorVersion);
  EXPECT_EQ(35, _ctx.getImageVersion().minorVersion);
}

TEST_F(WinLinkParserTest, MinMajorOSVersion) {
  EXPECT_TRUE(parse("link.exe", "/subsystem:windows,3", "foo.o", nullptr));
  EXPECT_EQ(llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_GUI, _ctx.getSubsystem());
  EXPECT_EQ(3, _ctx.getMinOSVersion().majorVersion);
  EXPECT_EQ(0, _ctx.getMinOSVersion().minorVersion);
}

TEST_F(WinLinkParserTest, MinMajorMinorOSVersion) {
  EXPECT_TRUE(parse("link.exe", "/subsystem:windows,3.1", "foo.o", nullptr));
  EXPECT_EQ(llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_GUI, _ctx.getSubsystem());
  EXPECT_EQ(3, _ctx.getMinOSVersion().majorVersion);
  EXPECT_EQ(1, _ctx.getMinOSVersion().minorVersion);
}

TEST_F(WinLinkParserTest, Base) {
  EXPECT_TRUE(parse("link.exe", "/base:8388608", "a.obj", nullptr));
  EXPECT_EQ(0x800000U, _ctx.getBaseAddress());
}

TEST_F(WinLinkParserTest, InvalidBase) {
  EXPECT_FALSE(parse("link.exe", "/base:1234", "a.obj", nullptr));
  EXPECT_TRUE(StringRef(errorMessage())
                  .startswith("Base address have to be multiple of 64K"));
}

TEST_F(WinLinkParserTest, StackReserve) {
  EXPECT_TRUE(parse("link.exe", "/stack:8192", "a.obj", nullptr));
  EXPECT_EQ(8192U, _ctx.getStackReserve());
  EXPECT_EQ(4096U, _ctx.getStackCommit());
}

TEST_F(WinLinkParserTest, StackReserveAndCommit) {
  EXPECT_TRUE(parse("link.exe", "/stack:16384,8192", "a.obj", nullptr));
  EXPECT_EQ(16384U, _ctx.getStackReserve());
  EXPECT_EQ(8192U, _ctx.getStackCommit());
}

TEST_F(WinLinkParserTest, InvalidStackSize) {
  EXPECT_FALSE(parse("link.exe", "/stack:8192,16384", "a.obj", nullptr));
  EXPECT_TRUE(StringRef(errorMessage()).startswith("Invalid stack size"));
}

TEST_F(WinLinkParserTest, HeapReserve) {
  EXPECT_TRUE(parse("link.exe", "/heap:8192", "a.obj", nullptr));
  EXPECT_EQ(8192U, _ctx.getHeapReserve());
  EXPECT_EQ(4096U, _ctx.getHeapCommit());
}

TEST_F(WinLinkParserTest, HeapReserveAndCommit) {
  EXPECT_TRUE(parse("link.exe", "/heap:16384,8192", "a.obj", nullptr));
  EXPECT_EQ(16384U, _ctx.getHeapReserve());
  EXPECT_EQ(8192U, _ctx.getHeapCommit());
}

TEST_F(WinLinkParserTest, InvalidHeapSize) {
  EXPECT_FALSE(parse("link.exe", "/heap:8192,16384", "a.obj", nullptr));
  EXPECT_TRUE(StringRef(errorMessage()).startswith("Invalid heap size"));
}

TEST_F(WinLinkParserTest, SectionAlignment) {
  EXPECT_TRUE(parse("link.exe", "/align:8192", "a.obj", nullptr));
  EXPECT_EQ(8192U, _ctx.getSectionDefaultAlignment());
}

TEST_F(WinLinkParserTest, InvalidAlignment) {
  EXPECT_FALSE(parse("link.exe", "/align:1000", "a.obj", nullptr));
  EXPECT_EQ("Section alignment must be a power of 2, but got 1000\n",
            errorMessage());
}

TEST_F(WinLinkParserTest, Include) {
  EXPECT_TRUE(parse("link.exe", "/include:foo", "a.out", nullptr));
  auto symbols = _ctx.initialUndefinedSymbols();
  EXPECT_FALSE(symbols.empty());
  EXPECT_EQ("foo", symbols[0]);
}

TEST_F(WinLinkParserTest, Merge) {
  EXPECT_TRUE(parse("link.exe", "/merge:.foo=.bar", "/merge:.bar=.baz",
                    "a.out", nullptr));
  EXPECT_EQ(".baz", _ctx.getOutputSectionName(".foo"));
  EXPECT_EQ(".baz", _ctx.getOutputSectionName(".bar"));
  EXPECT_EQ(".abc", _ctx.getOutputSectionName(".abc"));
}

TEST_F(WinLinkParserTest, Merge_Circular) {
  EXPECT_FALSE(parse("link.exe", "/merge:.foo=.bar", "/merge:.bar=.foo",
                     "a.out", nullptr));
}

TEST_F(WinLinkParserTest, Implib) {
  EXPECT_TRUE(parse("link.exe", "/implib:foo.dll.lib", "a.out", nullptr));
  EXPECT_EQ("foo.dll.lib", _ctx.getOutputImportLibraryPath());
}

TEST_F(WinLinkParserTest, ImplibDefault) {
  EXPECT_TRUE(parse("link.exe", "/out:foobar.dll", "a.out", nullptr));
  EXPECT_EQ("foobar.lib", _ctx.getOutputImportLibraryPath());
}

//
// Tests for /section
//

namespace {
const uint32_t discardable = llvm::COFF::IMAGE_SCN_MEM_DISCARDABLE;
const uint32_t not_cached = llvm::COFF::IMAGE_SCN_MEM_NOT_CACHED;
const uint32_t not_paged = llvm::COFF::IMAGE_SCN_MEM_NOT_PAGED;
const uint32_t shared = llvm::COFF::IMAGE_SCN_MEM_SHARED;
const uint32_t execute = llvm::COFF::IMAGE_SCN_MEM_EXECUTE;
const uint32_t read = llvm::COFF::IMAGE_SCN_MEM_READ;
const uint32_t write = llvm::COFF::IMAGE_SCN_MEM_WRITE;

#define TEST_SECTION(testname, arg, expect)                                    \
  TEST_F(WinLinkParserTest, testname) {                                        \
    EXPECT_TRUE(parse("link.exe", "/section:.text," arg, "a.obj", nullptr));   \
    EXPECT_EQ(expect, _ctx.getSectionAttributes(".text", execute | read)); \
  }

TEST_SECTION(SectionD, "d", execute | read | discardable)
TEST_SECTION(SectionE, "e", execute)
TEST_SECTION(SectionK, "k", execute | read | not_cached)
TEST_SECTION(SectionP, "p", execute | read | not_paged)
TEST_SECTION(SectionR, "r", read)
TEST_SECTION(SectionS, "s", execute | read | shared)
TEST_SECTION(SectionW, "w", write)

#undef TEST_SECTION

TEST_F(WinLinkParserTest, Section) {
  EXPECT_TRUE(parse("link.exe", "/section:.text,dekprsw",
                    "/section:.text,!dekprsw", "a.obj", nullptr));
  EXPECT_EQ(0U, _ctx.getSectionAttributes(".text", execute | read));
}

TEST_F(WinLinkParserTest, SectionNegate) {
  EXPECT_TRUE(parse("link.exe", "/section:.text,!e", "a.obj", nullptr));
  EXPECT_EQ(read, _ctx.getSectionAttributes(".text", execute | read));
}

TEST_F(WinLinkParserTest, SectionMultiple) {
  EXPECT_TRUE(parse("link.exe", "/section:.foo,e", "/section:.foo,rw",
                    "/section:.foo,!d", "a.obj", nullptr));
  uint32_t flags = execute | read | not_paged | discardable;
  uint32_t expected = execute | read | write | not_paged;
  EXPECT_EQ(expected, _ctx.getSectionAttributes(".foo", flags));
}

} // end anonymous namespace

//
// Tests for /defaultlib and /nodefaultlib.
//

TEST_F(WinLinkParserTest, DefaultLib) {
  EXPECT_TRUE(parse("link.exe", "/defaultlib:user32.lib",
                    "/defaultlib:kernel32", "a.obj", nullptr));
  EXPECT_EQ(4, inputFileCount());
  EXPECT_EQ("a.obj", inputFile(0));
  EXPECT_EQ("user32.lib", inputFile(1));
  EXPECT_EQ("kernel32.lib", inputFile(2));
}

TEST_F(WinLinkParserTest, DefaultLibDuplicates) {
  EXPECT_TRUE(parse("link.exe", "/defaultlib:user32.lib",
                    "/defaultlib:user32.lib", "a.obj", nullptr));
  EXPECT_EQ(3, inputFileCount());
  EXPECT_EQ("a.obj", inputFile(0));
  EXPECT_EQ("user32.lib", inputFile(1));
}

TEST_F(WinLinkParserTest, NoDefaultLib) {
  EXPECT_TRUE(parse("link.exe", "/defaultlib:user32.lib",
                    "/defaultlib:kernel32", "/nodefaultlib:user32.lib", "a.obj",
                    nullptr));
  EXPECT_EQ(3, inputFileCount());
  EXPECT_EQ("a.obj", inputFile(0));
  EXPECT_EQ("kernel32.lib", inputFile(1));
}

TEST_F(WinLinkParserTest, NoDefaultLibCase) {
  EXPECT_TRUE(parse("link.exe", "/defaultlib:user32",
                    "/defaultlib:kernel32", "/nodefaultlib:USER32.LIB", "a.obj",
                    nullptr));
  EXPECT_EQ(3, inputFileCount());
  EXPECT_EQ("a.obj", inputFile(0));
  EXPECT_EQ("kernel32.lib", inputFile(1));
}

TEST_F(WinLinkParserTest, NoDefaultLibAll) {
  EXPECT_TRUE(parse("link.exe", "/defaultlib:user32.lib",
                    "/defaultlib:kernel32", "/nodefaultlib", "a.obj", nullptr));
  EXPECT_EQ(2, inputFileCount());
  EXPECT_EQ("a.obj", inputFile(0));
}

TEST_F(WinLinkParserTest, DisallowLib) {
  EXPECT_TRUE(parse("link.exe", "/defaultlib:user32.lib",
                    "/defaultlib:kernel32", "/disallowlib:user32.lib", "a.obj",
                    nullptr));
  EXPECT_EQ(3, inputFileCount());
  EXPECT_EQ("a.obj", inputFile(0));
  EXPECT_EQ("kernel32.lib", inputFile(1));
}

//
// Tests for DLL.
//

TEST_F(WinLinkParserTest, NoEntry) {
  EXPECT_TRUE(parse("link.exe", "/noentry", "/dll", "a.obj", nullptr));
  EXPECT_TRUE(_ctx.isDll());
  EXPECT_EQ(0x10000000U, _ctx.getBaseAddress());
  EXPECT_EQ("", _ctx.entrySymbolName());
}

TEST_F(WinLinkParserTest, NoEntryError) {
  // /noentry without /dll is an error.
  EXPECT_FALSE(parse("link.exe", "/noentry", "a.obj", nullptr));
  EXPECT_EQ("/noentry must be specified with /dll\n", errorMessage());
}

//
// Tests for DELAYLOAD.
//

TEST_F(WinLinkParserTest, DelayLoad) {
  EXPECT_TRUE(parse("link.exe", "/delayload:abc.dll", "/delayload:def.dll",
                    "a.obj", nullptr));
  EXPECT_TRUE(_ctx.isDelayLoadDLL("abc.dll"));
  EXPECT_TRUE(_ctx.isDelayLoadDLL("DEF.DLL"));
  EXPECT_FALSE(_ctx.isDelayLoadDLL("xyz.dll"));
}

//
// Tests for SEH.
//

TEST_F(WinLinkParserTest, SafeSEH) {
  EXPECT_TRUE(parse("link.exe", "/safeseh", "a.obj", nullptr));
  EXPECT_TRUE(_ctx.requireSEH());
  EXPECT_FALSE(_ctx.noSEH());
}

TEST_F(WinLinkParserTest, NoSafeSEH) {
  EXPECT_TRUE(parse("link.exe", "/safeseh:no", "a.obj", nullptr));
  EXPECT_FALSE(_ctx.requireSEH());
  EXPECT_TRUE(_ctx.noSEH());
}

//
// Tests for boolean flags.
//

TEST_F(WinLinkParserTest, Force) {
  EXPECT_TRUE(parse("link.exe", "/force", "a.obj", nullptr));
  EXPECT_TRUE(_ctx.allowRemainingUndefines());
}

TEST_F(WinLinkParserTest, ForceUnresolved) {
  EXPECT_TRUE(parse("link.exe", "/force:unresolved", "a.obj", nullptr));
  EXPECT_TRUE(_ctx.allowRemainingUndefines());
}

TEST_F(WinLinkParserTest, NoNxCompat) {
  EXPECT_TRUE(parse("link.exe", "/nxcompat:no", "a.obj", nullptr));
  EXPECT_FALSE(_ctx.isNxCompat());
}

TEST_F(WinLinkParserTest, LargeAddressAware) {
  EXPECT_TRUE(parse("link.exe", "/largeaddressaware", "a.obj", nullptr));
  EXPECT_TRUE(_ctx.getLargeAddressAware());
}

TEST_F(WinLinkParserTest, NoLargeAddressAware) {
  EXPECT_TRUE(parse("link.exe", "/largeaddressaware:no", "a.obj", nullptr));
  EXPECT_FALSE(_ctx.getLargeAddressAware());
}

TEST_F(WinLinkParserTest, AllowBind) {
  EXPECT_TRUE(parse("link.exe", "/allowbind", "a.obj", nullptr));
  EXPECT_TRUE(_ctx.getAllowBind());
}

TEST_F(WinLinkParserTest, NoAllowBind) {
  EXPECT_TRUE(parse("link.exe", "/allowbind:no", "a.obj", nullptr));
  EXPECT_FALSE(_ctx.getAllowBind());
}

TEST_F(WinLinkParserTest, AllowIsolation) {
  EXPECT_TRUE(parse("link.exe", "/allowisolation", "a.obj", nullptr));
  EXPECT_TRUE(_ctx.getAllowIsolation());
}

TEST_F(WinLinkParserTest, NoAllowIsolation) {
  EXPECT_TRUE(parse("link.exe", "/allowisolation:no", "a.obj", nullptr));
  EXPECT_FALSE(_ctx.getAllowIsolation());
}

TEST_F(WinLinkParserTest, SwapRunFromCD) {
  EXPECT_TRUE(parse("link.exe", "/swaprun:cd", "a.obj", nullptr));
  EXPECT_TRUE(_ctx.getSwapRunFromCD());
}

TEST_F(WinLinkParserTest, SwapRunFromNet) {
  EXPECT_TRUE(parse("link.exe", "/swaprun:net", "a.obj", nullptr));
  EXPECT_TRUE(_ctx.getSwapRunFromNet());
}

TEST_F(WinLinkParserTest, Debug) {
  EXPECT_TRUE(parse("link.exe", "/debug", "a.obj", nullptr));
  EXPECT_TRUE(_ctx.deadStrip());
  EXPECT_TRUE(_ctx.getDebug());
  EXPECT_EQ("a.pdb", _ctx.getPDBFilePath());
}

TEST_F(WinLinkParserTest, PDB) {
  EXPECT_TRUE(parse("link.exe", "/debug", "/pdb:foo.pdb", "a.obj", nullptr));
  EXPECT_TRUE(_ctx.getDebug());
  EXPECT_EQ("foo.pdb", _ctx.getPDBFilePath());
}

TEST_F(WinLinkParserTest, Fixed) {
  EXPECT_TRUE(parse("link.exe", "/fixed", "a.out", nullptr));
  EXPECT_FALSE(_ctx.getBaseRelocationEnabled());
  EXPECT_FALSE(_ctx.getDynamicBaseEnabled());
}

TEST_F(WinLinkParserTest, NoFixed) {
  EXPECT_TRUE(parse("link.exe", "/fixed:no", "a.out", nullptr));
  EXPECT_TRUE(_ctx.getBaseRelocationEnabled());
}

TEST_F(WinLinkParserTest, TerminalServerAware) {
  EXPECT_TRUE(parse("link.exe", "/tsaware", "a.out", nullptr));
  EXPECT_TRUE(_ctx.isTerminalServerAware());
}

TEST_F(WinLinkParserTest, NoTerminalServerAware) {
  EXPECT_TRUE(parse("link.exe", "/tsaware:no", "a.out", nullptr));
  EXPECT_FALSE(_ctx.isTerminalServerAware());
}

TEST_F(WinLinkParserTest, DynamicBase) {
  EXPECT_TRUE(parse("link.exe", "/dynamicbase", "a.out", nullptr));
  EXPECT_TRUE(_ctx.getDynamicBaseEnabled());
}

TEST_F(WinLinkParserTest, NoDynamicBase) {
  EXPECT_TRUE(parse("link.exe", "/dynamicbase:no", "a.out", nullptr));
  EXPECT_FALSE(_ctx.getDynamicBaseEnabled());
}

//
// Test for /failifmismatch
//

TEST_F(WinLinkParserTest, FailIfMismatch_Match) {
  EXPECT_TRUE(parse("link.exe", "/failifmismatch:foo=bar",
                    "/failifmismatch:foo=bar", "/failifmismatch:abc=def",
                    "a.out", nullptr));
}

TEST_F(WinLinkParserTest, FailIfMismatch_Mismatch) {
  EXPECT_FALSE(parse("link.exe", "/failifmismatch:foo=bar",
                     "/failifmismatch:foo=baz", "a.out", nullptr));
}

//
// Tests for /manifest, /manifestuac, /manifestfile, and /manifestdependency.
//
TEST_F(WinLinkParserTest, Manifest_Default) {
  EXPECT_TRUE(parse("link.exe", "/manifest", "a.out", nullptr));
  EXPECT_TRUE(_ctx.getCreateManifest());
  EXPECT_FALSE(_ctx.getEmbedManifest());
  EXPECT_EQ(1, _ctx.getManifestId());
  EXPECT_EQ("'asInvoker'", _ctx.getManifestLevel());
  EXPECT_EQ("'false'", _ctx.getManifestUiAccess());
}

TEST_F(WinLinkParserTest, Manifest_No) {
  EXPECT_TRUE(parse("link.exe", "/manifest:no", "a.out", nullptr));
  EXPECT_FALSE(_ctx.getCreateManifest());
}

TEST_F(WinLinkParserTest, Manifestuac_no) {
  EXPECT_TRUE(parse("link.exe", "/manifestuac:NO", "a.out", nullptr));
  EXPECT_FALSE(_ctx.getManifestUAC());
}

TEST_F(WinLinkParserTest, Manifestuac_Level) {
  EXPECT_TRUE(parse("link.exe", "/manifestuac:level='requireAdministrator'",
                    "a.out", nullptr));
  EXPECT_EQ("'requireAdministrator'", _ctx.getManifestLevel());
  EXPECT_EQ("'false'", _ctx.getManifestUiAccess());
}

TEST_F(WinLinkParserTest, Manifestuac_UiAccess) {
  EXPECT_TRUE(parse("link.exe", "/manifestuac:uiAccess='true'", "a.out", nullptr));
  EXPECT_EQ("'asInvoker'", _ctx.getManifestLevel());
  EXPECT_EQ("'true'", _ctx.getManifestUiAccess());
}

TEST_F(WinLinkParserTest, Manifestuac_LevelAndUiAccess) {
  EXPECT_TRUE(parse("link.exe",
                    "/manifestuac:level='requireAdministrator' uiAccess='true'",
                    "a.out", nullptr));
  EXPECT_EQ("'requireAdministrator'", _ctx.getManifestLevel());
  EXPECT_EQ("'true'", _ctx.getManifestUiAccess());
}

TEST_F(WinLinkParserTest, Manifestfile) {
  EXPECT_TRUE(parse("link.exe", "/manifestfile:bar.manifest",
                    "a.out", nullptr));
  EXPECT_EQ("bar.manifest", _ctx.getManifestOutputPath());
}

TEST_F(WinLinkParserTest, Manifestdependency) {
  EXPECT_TRUE(parse("link.exe", "/manifestdependency:foo bar", "a.out",
                    nullptr));
  EXPECT_EQ("foo bar", _ctx.getManifestDependency());
}

//
// Test for /OPT
//

TEST_F(WinLinkParserTest, OptNoRef) {
  EXPECT_TRUE(parse("link.exe", "/opt:noref", "a.obj", nullptr));
  EXPECT_FALSE(_ctx.deadStrip());
}

TEST_F(WinLinkParserTest, OptIgnore) {
  EXPECT_TRUE(parse("link.exe", "/opt:ref", "/opt:icf", "/opt:noicf",
                    "/opt:icf=foo", "/opt:lbr", "/opt:nolbr", "a.obj",
                    nullptr));
}

TEST_F(WinLinkParserTest, OptUnknown) {
  EXPECT_FALSE(parse("link.exe", "/opt:foo", "a.obj", nullptr));
}

//
// Test for /PROFILE
//

TEST_F(WinLinkParserTest, Profile) {
  EXPECT_TRUE(parse("link.exe", "/profile", "a.obj", nullptr));
  EXPECT_TRUE(_ctx.deadStrip());
  EXPECT_TRUE(_ctx.getBaseRelocationEnabled());
  EXPECT_TRUE(_ctx.getDynamicBaseEnabled());
}

//
// Test for command line flags that are ignored.
//

TEST_F(WinLinkParserTest, Ignore) {
  // There are some no-op command line options that are recognized for
  // compatibility with link.exe.
  EXPECT_TRUE(parse("link.exe", "/nologo", "/errorreport:prompt",
                    "/incremental", "/incremental:no", "/delay:unload",
                    "/disallowlib:foo", "/pdbaltpath:bar",
                    "/wx", "/wx:no", "/tlbid:1", "/tlbout:foo", "/idlout:foo",
                    "/ignore:4000", "/ignoreidl", "/implib:foo", "/safeseh",
                    "/safeseh:no", "/functionpadmin", "/maxilksize:1024",
                    "a.obj", nullptr));
  EXPECT_EQ("", errorMessage());
  EXPECT_EQ(2, inputFileCount());
  EXPECT_EQ("a.obj", inputFile(0));
}

//
// Test for "--"
//

TEST_F(WinLinkParserTest, DashDash) {
  EXPECT_TRUE(parse("link.exe", "/subsystem:console", "/out:a.exe", "a.obj",
                    "--", "b.obj", "-c.obj", nullptr));
  EXPECT_EQ(llvm::COFF::IMAGE_SUBSYSTEM_WINDOWS_CUI, _ctx.getSubsystem());
  EXPECT_EQ("a.exe", _ctx.outputPath());
  EXPECT_EQ(4, inputFileCount());
  EXPECT_EQ("a.obj", inputFile(0));
  EXPECT_EQ("b.obj", inputFile(1));
  EXPECT_EQ("-c.obj", inputFile(2));
}
