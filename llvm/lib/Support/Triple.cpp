//===--- Triple.cpp - Target triple helper class --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Triple.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstring>
using namespace llvm;

const char *Triple::getArchTypeName(ArchType Kind) {
  switch (Kind) {
  case UnknownArch: return "unknown";

  case arm:     return "arm";
  case cellspu: return "cellspu";
  case hexagon: return "hexagon";
  case mips:    return "mips";
  case mipsel:  return "mipsel";
  case mips64:  return "mips64";
  case mips64el:return "mips64el";
  case msp430:  return "msp430";
  case ppc64:   return "powerpc64";
  case ppc:     return "powerpc";
  case sparc:   return "sparc";
  case sparcv9: return "sparcv9";
  case tce:     return "tce";
  case thumb:   return "thumb";
  case x86:     return "i386";
  case x86_64:  return "x86_64";
  case xcore:   return "xcore";
  case mblaze:  return "mblaze";
  case ptx32:   return "ptx32";
  case ptx64:   return "ptx64";
  case le32:    return "le32";
  case amdil:   return "amdil";
  }

  llvm_unreachable("Invalid ArchType!");
}

const char *Triple::getArchTypePrefix(ArchType Kind) {
  switch (Kind) {
  default:
    return 0;

  case arm:
  case thumb:   return "arm";

  case cellspu: return "spu";

  case ppc64:
  case ppc:     return "ppc";

  case mblaze:  return "mblaze";

  case hexagon:   return "hexagon";

  case sparcv9:
  case sparc:   return "sparc";

  case x86:
  case x86_64:  return "x86";

  case xcore:   return "xcore";

  case ptx32:   return "ptx";
  case ptx64:   return "ptx";
  case le32:    return "le32";
  case amdil:   return "amdil";
  }
}

const char *Triple::getVendorTypeName(VendorType Kind) {
  switch (Kind) {
  case UnknownVendor: return "unknown";

  case Apple: return "apple";
  case PC: return "pc";
  case SCEI: return "scei";
  }

  llvm_unreachable("Invalid VendorType!");
}

const char *Triple::getOSTypeName(OSType Kind) {
  switch (Kind) {
  case UnknownOS: return "unknown";

  case AuroraUX: return "auroraux";
  case Cygwin: return "cygwin";
  case Darwin: return "darwin";
  case DragonFly: return "dragonfly";
  case FreeBSD: return "freebsd";
  case IOS: return "ios";
  case KFreeBSD: return "kfreebsd";
  case Linux: return "linux";
  case Lv2: return "lv2";
  case MacOSX: return "macosx";
  case MinGW32: return "mingw32";
  case NetBSD: return "netbsd";
  case OpenBSD: return "openbsd";
  case Psp: return "psp";
  case Solaris: return "solaris";
  case Win32: return "win32";
  case Haiku: return "haiku";
  case Minix: return "minix";
  case RTEMS: return "rtems";
  case NativeClient: return "nacl";
  }

  llvm_unreachable("Invalid OSType");
}

const char *Triple::getEnvironmentTypeName(EnvironmentType Kind) {
  switch (Kind) {
  case UnknownEnvironment: return "unknown";
  case GNU: return "gnu";
  case GNUEABIHF: return "gnueabihf";
  case GNUEABI: return "gnueabi";
  case EABI: return "eabi";
  case MachO: return "macho";
  case ANDROIDEABI: return "androideabi";
  }

  llvm_unreachable("Invalid EnvironmentType!");
}

Triple::ArchType Triple::getArchTypeForLLVMName(StringRef Name) {
  return StringSwitch<Triple::ArchType>(Name)
    .Case("arm", arm)
    .Case("cellspu", cellspu)
    .Case("mips", mips)
    .Case("mipsel", mipsel)
    .Case("mips64", mips64)
    .Case("mips64el", mips64el)
    .Case("msp430", msp430)
    .Case("ppc64", ppc64)
    .Case("ppc32", ppc)
    .Case("ppc", ppc)
    .Case("mblaze", mblaze)
    .Case("hexagon", hexagon)
    .Case("sparc", sparc)
    .Case("sparcv9", sparcv9)
    .Case("tce", tce)
    .Case("thumb", thumb)
    .Case("x86", x86)
    .Case("x86-64", x86_64)
    .Case("xcore", xcore)
    .Case("ptx32", ptx32)
    .Case("ptx64", ptx64)
    .Case("le32", le32)
    .Case("amdil", amdil)
    .Default(UnknownArch);
}

Triple::ArchType Triple::getArchTypeForDarwinArchName(StringRef Str) {
  // See arch(3) and llvm-gcc's driver-driver.c. We don't implement support for
  // archs which Darwin doesn't use.

  // The matching this routine does is fairly pointless, since it is neither the
  // complete architecture list, nor a reasonable subset. The problem is that
  // historically the driver driver accepts this and also ties its -march=
  // handling to the architecture name, so we need to be careful before removing
  // support for it.

  // This code must be kept in sync with Clang's Darwin specific argument
  // translation.

  return StringSwitch<ArchType>(Str)
    .Cases("ppc", "ppc601", "ppc603", "ppc604", "ppc604e", Triple::ppc)
    .Cases("ppc750", "ppc7400", "ppc7450", "ppc970", Triple::ppc)
    .Case("ppc64", Triple::ppc64)
    .Cases("i386", "i486", "i486SX", "i586", "i686", Triple::x86)
    .Cases("pentium", "pentpro", "pentIIm3", "pentIIm5", "pentium4",
           Triple::x86)
    .Case("x86_64", Triple::x86_64)
    // This is derived from the driver driver.
    .Cases("arm", "armv4t", "armv5", "armv6", Triple::arm)
    .Cases("armv7", "armv7f", "armv7k", "armv7s", "xscale", Triple::arm)
    .Case("ptx32", Triple::ptx32)
    .Case("ptx64", Triple::ptx64)
    .Case("amdil", Triple::amdil)
    .Default(Triple::UnknownArch);
}

// Returns architecture name that is understood by the target assembler.
const char *Triple::getArchNameForAssembler() {
  if (!isOSDarwin() && getVendor() != Triple::Apple)
    return NULL;

  return StringSwitch<const char*>(getArchName())
    .Case("i386", "i386")
    .Case("x86_64", "x86_64")
    .Case("powerpc", "ppc")
    .Case("powerpc64", "ppc64")
    .Cases("mblaze", "microblaze", "mblaze")
    .Case("arm", "arm")
    .Cases("armv4t", "thumbv4t", "armv4t")
    .Cases("armv5", "armv5e", "thumbv5", "thumbv5e", "armv5")
    .Cases("armv6", "thumbv6", "armv6")
    .Cases("armv7", "thumbv7", "armv7")
    .Case("ptx32", "ptx32")
    .Case("ptx64", "ptx64")
    .Case("le32", "le32")
    .Case("amdil", "amdil")
    .Default(NULL);
}

Triple::ArchType Triple::ParseArch(StringRef ArchName) {
  return StringSwitch<ArchType>(ArchName)
    .Cases("i386", "i486", "i586", "i686", x86)
    .Cases("i786", "i886", "i986", x86) // FIXME: Do we need to support these?
    .Cases("amd64", "x86_64", x86_64)
    .Case("powerpc", ppc)
    .Cases("powerpc64", "ppu", ppc64)
    .Case("mblaze", mblaze)
    .Cases("arm", "xscale", arm)
    // FIXME: It would be good to replace these with explicit names for all the
    // various suffixes supported.
    .StartsWith("armv", arm)
    .Case("thumb", thumb)
    .StartsWith("thumbv", thumb)
    .Cases("spu", "cellspu", cellspu)
    .Case("msp430", msp430)
    .Cases("mips", "mipseb", "mipsallegrex", mips)
    .Cases("mipsel", "mipsallegrexel", "psp", mipsel)
    .Cases("mips64", "mips64eb", mips64)
    .Case("mips64el", mips64el)
    .Case("hexagon", hexagon)
    .Case("sparc", sparc)
    .Case("sparcv9", sparcv9)
    .Case("tce", tce)
    .Case("xcore", xcore)
    .Case("ptx32", ptx32)
    .Case("ptx64", ptx64)
    .Case("le32", le32)
    .Case("amdil", amdil)
    .Default(UnknownArch);
}

Triple::VendorType Triple::ParseVendor(StringRef VendorName) {
  return StringSwitch<VendorType>(VendorName)
    .Case("apple", Apple)
    .Case("pc", PC)
    .Case("scei", SCEI)
    .Default(UnknownVendor);
}

Triple::OSType Triple::ParseOS(StringRef OSName) {
  return StringSwitch<OSType>(OSName)
    .StartsWith("auroraux", AuroraUX)
    .StartsWith("cygwin", Cygwin)
    .StartsWith("darwin", Darwin)
    .StartsWith("dragonfly", DragonFly)
    .StartsWith("freebsd", FreeBSD)
    .StartsWith("ios", IOS)
    .StartsWith("kfreebsd", KFreeBSD)
    .StartsWith("linux", Linux)
    .StartsWith("lv2", Lv2)
    .StartsWith("macosx", MacOSX)
    .StartsWith("mingw32", MinGW32)
    .StartsWith("netbsd", NetBSD)
    .StartsWith("openbsd", OpenBSD)
    .StartsWith("psp", Psp)
    .StartsWith("solaris", Solaris)
    .StartsWith("win32", Win32)
    .StartsWith("haiku", Haiku)
    .StartsWith("minix", Minix)
    .StartsWith("rtems", RTEMS)
    .StartsWith("nacl", NativeClient)
    .Default(UnknownOS);
}

Triple::EnvironmentType Triple::ParseEnvironment(StringRef EnvironmentName) {
  return StringSwitch<EnvironmentType>(EnvironmentName)
    .StartsWith("eabi", EABI)
    .StartsWith("gnueabihf", GNUEABIHF)
    .StartsWith("gnueabi", GNUEABI)
    .StartsWith("gnu", GNU)
    .StartsWith("macho", MachO)
    .StartsWith("androideabi", ANDROIDEABI)
    .Default(UnknownEnvironment);
}

/// \brief Construct a triple from the string representation provided.
///
/// This doesn't actually parse the string representation eagerly. Instead it
/// stores it, and tracks the fact that it hasn't been parsed. The first time
/// any of the structural queries are made, the string is parsed and the
/// results cached in various members.
Triple::Triple(const Twine &Str)
    : Data(Str.str()),
      Arch(ParseArch(getArchName())),
      Vendor(ParseVendor(getVendorName())),
      OS(ParseOS(getOSName())),
      Environment(ParseEnvironment(getEnvironmentName())) {
}

/// \brief Construct a triple from string representations of the architecture,
/// vendor, and OS.
///
/// This doesn't actually use these already distinct strings to setup the
/// triple information. Instead it joins them into a canonical form of a triple
/// string, and lazily parses it on use.
Triple::Triple(const Twine &ArchStr, const Twine &VendorStr, const Twine &OSStr)
    : Data((ArchStr + Twine('-') + VendorStr + Twine('-') + OSStr).str()),
      Arch(ParseArch(ArchStr.str())),
      Vendor(ParseVendor(VendorStr.str())),
      OS(ParseOS(OSStr.str())),
      Environment() {
}

/// \brief Construct a triple from string representations of the architecture,
/// vendor, OS, and environment.
///
/// This doesn't actually use these already distinct strings to setup the
/// triple information. Instead it joins them into a canonical form of a triple
/// string, and lazily parses it on use.
Triple::Triple(const Twine &ArchStr, const Twine &VendorStr, const Twine &OSStr,
               const Twine &EnvironmentStr)
    : Data((ArchStr + Twine('-') + VendorStr + Twine('-') + OSStr + Twine('-') +
            EnvironmentStr).str()),
      Arch(ParseArch(ArchStr.str())),
      Vendor(ParseVendor(VendorStr.str())),
      OS(ParseOS(OSStr.str())),
      Environment(ParseEnvironment(EnvironmentStr.str())) {
}

std::string Triple::normalize(StringRef Str) {
  // Parse into components.
  SmallVector<StringRef, 4> Components;
  for (size_t First = 0, Last = 0; Last != StringRef::npos; First = Last + 1) {
    Last = Str.find('-', First);
    Components.push_back(Str.slice(First, Last));
  }

  // If the first component corresponds to a known architecture, preferentially
  // use it for the architecture.  If the second component corresponds to a
  // known vendor, preferentially use it for the vendor, etc.  This avoids silly
  // component movement when a component parses as (eg) both a valid arch and a
  // valid os.
  ArchType Arch = UnknownArch;
  if (Components.size() > 0)
    Arch = ParseArch(Components[0]);
  VendorType Vendor = UnknownVendor;
  if (Components.size() > 1)
    Vendor = ParseVendor(Components[1]);
  OSType OS = UnknownOS;
  if (Components.size() > 2)
    OS = ParseOS(Components[2]);
  EnvironmentType Environment = UnknownEnvironment;
  if (Components.size() > 3)
    Environment = ParseEnvironment(Components[3]);

  // Note which components are already in their final position.  These will not
  // be moved.
  bool Found[4];
  Found[0] = Arch != UnknownArch;
  Found[1] = Vendor != UnknownVendor;
  Found[2] = OS != UnknownOS;
  Found[3] = Environment != UnknownEnvironment;

  // If they are not there already, permute the components into their canonical
  // positions by seeing if they parse as a valid architecture, and if so moving
  // the component to the architecture position etc.
  for (unsigned Pos = 0; Pos != array_lengthof(Found); ++Pos) {
    if (Found[Pos])
      continue; // Already in the canonical position.

    for (unsigned Idx = 0; Idx != Components.size(); ++Idx) {
      // Do not reparse any components that already matched.
      if (Idx < array_lengthof(Found) && Found[Idx])
        continue;

      // Does this component parse as valid for the target position?
      bool Valid = false;
      StringRef Comp = Components[Idx];
      switch (Pos) {
      default: llvm_unreachable("unexpected component type!");
      case 0:
        Arch = ParseArch(Comp);
        Valid = Arch != UnknownArch;
        break;
      case 1:
        Vendor = ParseVendor(Comp);
        Valid = Vendor != UnknownVendor;
        break;
      case 2:
        OS = ParseOS(Comp);
        Valid = OS != UnknownOS;
        break;
      case 3:
        Environment = ParseEnvironment(Comp);
        Valid = Environment != UnknownEnvironment;
        break;
      }
      if (!Valid)
        continue; // Nope, try the next component.

      // Move the component to the target position, pushing any non-fixed
      // components that are in the way to the right.  This tends to give
      // good results in the common cases of a forgotten vendor component
      // or a wrongly positioned environment.
      if (Pos < Idx) {
        // Insert left, pushing the existing components to the right.  For
        // example, a-b-i386 -> i386-a-b when moving i386 to the front.
        StringRef CurrentComponent(""); // The empty component.
        // Replace the component we are moving with an empty component.
        std::swap(CurrentComponent, Components[Idx]);
        // Insert the component being moved at Pos, displacing any existing
        // components to the right.
        for (unsigned i = Pos; !CurrentComponent.empty(); ++i) {
          // Skip over any fixed components.
          while (i < array_lengthof(Found) && Found[i]) ++i;
          // Place the component at the new position, getting the component
          // that was at this position - it will be moved right.
          std::swap(CurrentComponent, Components[i]);
        }
      } else if (Pos > Idx) {
        // Push right by inserting empty components until the component at Idx
        // reaches the target position Pos.  For example, pc-a -> -pc-a when
        // moving pc to the second position.
        do {
          // Insert one empty component at Idx.
          StringRef CurrentComponent(""); // The empty component.
          for (unsigned i = Idx; i < Components.size();) {
            // Place the component at the new position, getting the component
            // that was at this position - it will be moved right.
            std::swap(CurrentComponent, Components[i]);
            // If it was placed on top of an empty component then we are done.
            if (CurrentComponent.empty())
              break;
            // Advance to the next component, skipping any fixed components.
            while (++i < array_lengthof(Found) && Found[i])
              ;
          }
          // The last component was pushed off the end - append it.
          if (!CurrentComponent.empty())
            Components.push_back(CurrentComponent);

          // Advance Idx to the component's new position.
          while (++Idx < array_lengthof(Found) && Found[Idx]) {}
        } while (Idx < Pos); // Add more until the final position is reached.
      }
      assert(Pos < Components.size() && Components[Pos] == Comp &&
             "Component moved wrong!");
      Found[Pos] = true;
      break;
    }
  }

  // Special case logic goes here.  At this point Arch, Vendor and OS have the
  // correct values for the computed components.

  // Stick the corrected components back together to form the normalized string.
  std::string Normalized;
  for (unsigned i = 0, e = Components.size(); i != e; ++i) {
    if (i) Normalized += '-';
    Normalized += Components[i];
  }
  return Normalized;
}

StringRef Triple::getArchName() const {
  return StringRef(Data).split('-').first;           // Isolate first component
}

StringRef Triple::getVendorName() const {
  StringRef Tmp = StringRef(Data).split('-').second; // Strip first component
  return Tmp.split('-').first;                       // Isolate second component
}

StringRef Triple::getOSName() const {
  StringRef Tmp = StringRef(Data).split('-').second; // Strip first component
  Tmp = Tmp.split('-').second;                       // Strip second component
  return Tmp.split('-').first;                       // Isolate third component
}

StringRef Triple::getEnvironmentName() const {
  StringRef Tmp = StringRef(Data).split('-').second; // Strip first component
  Tmp = Tmp.split('-').second;                       // Strip second component
  return Tmp.split('-').second;                      // Strip third component
}

StringRef Triple::getOSAndEnvironmentName() const {
  StringRef Tmp = StringRef(Data).split('-').second; // Strip first component
  return Tmp.split('-').second;                      // Strip second component
}

static unsigned EatNumber(StringRef &Str) {
  assert(!Str.empty() && Str[0] >= '0' && Str[0] <= '9' && "Not a number");
  unsigned Result = 0;

  do {
    // Consume the leading digit.
    Result = Result*10 + (Str[0] - '0');

    // Eat the digit.
    Str = Str.substr(1);
  } while (!Str.empty() && Str[0] >= '0' && Str[0] <= '9');

  return Result;
}

void Triple::getOSVersion(unsigned &Major, unsigned &Minor,
                          unsigned &Micro) const {
  StringRef OSName = getOSName();

  // Assume that the OS portion of the triple starts with the canonical name.
  StringRef OSTypeName = getOSTypeName(getOS());
  if (OSName.startswith(OSTypeName))
    OSName = OSName.substr(OSTypeName.size());

  // Any unset version defaults to 0.
  Major = Minor = Micro = 0;

  // Parse up to three components.
  unsigned *Components[3] = { &Major, &Minor, &Micro };
  for (unsigned i = 0; i != 3; ++i) {
    if (OSName.empty() || OSName[0] < '0' || OSName[0] > '9')
      break;

    // Consume the leading number.
    *Components[i] = EatNumber(OSName);

    // Consume the separator, if present.
    if (OSName.startswith("."))
      OSName = OSName.substr(1);
  }
}

bool Triple::getMacOSXVersion(unsigned &Major, unsigned &Minor,
                              unsigned &Micro) const {
  getOSVersion(Major, Minor, Micro);

  switch (getOS()) {
  default: llvm_unreachable("unexpected OS for Darwin triple");
  case Darwin:
    // Default to darwin8, i.e., MacOSX 10.4.
    if (Major == 0)
      Major = 8;
    // Darwin version numbers are skewed from OS X versions.
    if (Major < 4)
      return false;
    Micro = 0;
    Minor = Major - 4;
    Major = 10;
    break;
  case MacOSX:
    // Default to 10.4.
    if (Major == 0) {
      Major = 10;
      Minor = 4;
    }
    if (Major != 10)
      return false;
    break;
  case IOS:
    // Ignore the version from the triple.  This is only handled because the
    // the clang driver combines OS X and IOS support into a common Darwin
    // toolchain that wants to know the OS X version number even when targeting
    // IOS.
    Major = 10;
    Minor = 4;
    Micro = 0;
    break;
  }
  return true;
}

void Triple::setTriple(const Twine &Str) {
  *this = Triple(Str);
}

void Triple::setArch(ArchType Kind) {
  setArchName(getArchTypeName(Kind));
}

void Triple::setVendor(VendorType Kind) {
  setVendorName(getVendorTypeName(Kind));
}

void Triple::setOS(OSType Kind) {
  setOSName(getOSTypeName(Kind));
}

void Triple::setEnvironment(EnvironmentType Kind) {
  setEnvironmentName(getEnvironmentTypeName(Kind));
}

void Triple::setArchName(StringRef Str) {
  // Work around a miscompilation bug for Twines in gcc 4.0.3.
  SmallString<64> Triple;
  Triple += Str;
  Triple += "-";
  Triple += getVendorName();
  Triple += "-";
  Triple += getOSAndEnvironmentName();
  setTriple(Triple.str());
}

void Triple::setVendorName(StringRef Str) {
  setTriple(getArchName() + "-" + Str + "-" + getOSAndEnvironmentName());
}

void Triple::setOSName(StringRef Str) {
  if (hasEnvironment())
    setTriple(getArchName() + "-" + getVendorName() + "-" + Str +
              "-" + getEnvironmentName());
  else
    setTriple(getArchName() + "-" + getVendorName() + "-" + Str);
}

void Triple::setEnvironmentName(StringRef Str) {
  setTriple(getArchName() + "-" + getVendorName() + "-" + getOSName() +
            "-" + Str);
}

void Triple::setOSAndEnvironmentName(StringRef Str) {
  setTriple(getArchName() + "-" + getVendorName() + "-" + Str);
}

static unsigned getArchPointerBitWidth(llvm::Triple::ArchType Arch) {
  switch (Arch) {
  case llvm::Triple::UnknownArch:
    return 0;

  case llvm::Triple::msp430:
    return 16;

  case llvm::Triple::amdil:
  case llvm::Triple::arm:
  case llvm::Triple::cellspu:
  case llvm::Triple::hexagon:
  case llvm::Triple::le32:
  case llvm::Triple::mblaze:
  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
  case llvm::Triple::ppc:
  case llvm::Triple::ptx32:
  case llvm::Triple::sparc:
  case llvm::Triple::tce:
  case llvm::Triple::thumb:
  case llvm::Triple::x86:
  case llvm::Triple::xcore:
    return 32;

  case llvm::Triple::mips64:
  case llvm::Triple::mips64el:
  case llvm::Triple::ppc64:
  case llvm::Triple::ptx64:
  case llvm::Triple::sparcv9:
  case llvm::Triple::x86_64:
    return 64;
  }
  llvm_unreachable("Invalid architecture value");
}

bool Triple::isArch64Bit() const {
  return getArchPointerBitWidth(getArch()) == 64;
}

bool Triple::isArch32Bit() const {
  return getArchPointerBitWidth(getArch()) == 32;
}

bool Triple::isArch16Bit() const {
  return getArchPointerBitWidth(getArch()) == 16;
}

Triple Triple::get32BitArchVariant() const {
  Triple T(*this);
  switch (getArch()) {
  case Triple::UnknownArch:
  case Triple::msp430:
    T.setArch(UnknownArch);
    break;

  case Triple::amdil:
  case Triple::arm:
  case Triple::cellspu:
  case Triple::hexagon:
  case Triple::le32:
  case Triple::mblaze:
  case Triple::mips:
  case Triple::mipsel:
  case Triple::ppc:
  case Triple::ptx32:
  case Triple::sparc:
  case Triple::tce:
  case Triple::thumb:
  case Triple::x86:
  case Triple::xcore:
    // Already 32-bit.
    break;

  case Triple::mips64:    T.setArch(Triple::mips);    break;
  case Triple::mips64el:  T.setArch(Triple::mipsel);  break;
  case Triple::ppc64:     T.setArch(Triple::ppc);   break;
  case Triple::ptx64:     T.setArch(Triple::ptx32);   break;
  case Triple::sparcv9:   T.setArch(Triple::sparc);   break;
  case Triple::x86_64:    T.setArch(Triple::x86);     break;
  }
  return T;
}

Triple Triple::get64BitArchVariant() const {
  Triple T(*this);
  switch (getArch()) {
  case Triple::UnknownArch:
  case Triple::amdil:
  case Triple::arm:
  case Triple::cellspu:
  case Triple::hexagon:
  case Triple::le32:
  case Triple::mblaze:
  case Triple::msp430:
  case Triple::tce:
  case Triple::thumb:
  case Triple::xcore:
    T.setArch(UnknownArch);
    break;

  case Triple::mips64:
  case Triple::mips64el:
  case Triple::ppc64:
  case Triple::ptx64:
  case Triple::sparcv9:
  case Triple::x86_64:
    // Already 64-bit.
    break;

  case Triple::mips:    T.setArch(Triple::mips64);    break;
  case Triple::mipsel:  T.setArch(Triple::mips64el);  break;
  case Triple::ppc:     T.setArch(Triple::ppc64);     break;
  case Triple::ptx32:   T.setArch(Triple::ptx64);     break;
  case Triple::sparc:   T.setArch(Triple::sparcv9);   break;
  case Triple::x86:     T.setArch(Triple::x86_64);    break;
  }
  return T;
}
