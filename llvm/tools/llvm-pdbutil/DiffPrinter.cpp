
#include "DiffPrinter.h"

#include "llvm/Support/FormatAdapters.h"

using namespace llvm;
using namespace llvm::pdb;

static void setColor(llvm::raw_ostream &OS, DiffResult Result) {
  switch (Result) {
  case DiffResult::IDENTICAL:
    OS.changeColor(raw_ostream::Colors::GREEN, false);
    break;
  case DiffResult::EQUIVALENT:
    OS.changeColor(raw_ostream::Colors::YELLOW, true);
    break;
  default:
    OS.changeColor(raw_ostream::Colors::RED, false);
    break;
  }
}

DiffPrinter::DiffPrinter(uint32_t Indent, StringRef Header,
                         uint32_t PropertyWidth, uint32_t FieldWidth,
                         raw_ostream &Stream)
    : Indent(Indent), PropertyWidth(PropertyWidth), FieldWidth(FieldWidth),
      OS(Stream) {
  printHeaderRow();
  printFullRow(Header);
}

DiffPrinter::~DiffPrinter() {}

void DiffPrinter::printFullRow(StringRef Text) {
  newLine();
  printField(Text, DiffResult::UNSPECIFIED, AlignStyle::Center,
             PropertyWidth + 1 + FieldWidth + 1 + FieldWidth);
  printSeparatorRow();
}

void DiffPrinter::printSeparatorRow() {
  newLine();
  OS << formatv("{0}", fmt_repeat('-', PropertyWidth));
  OS << '+';
  OS << formatv("{0}", fmt_repeat('-', FieldWidth));
  OS << '+';
  OS << formatv("{0}", fmt_repeat('-', FieldWidth));
  OS << '|';
}

void DiffPrinter::printHeaderRow() {
  newLine('-');
  OS << formatv("{0}", fmt_repeat('-', PropertyWidth + 2 * FieldWidth + 3));
}

void DiffPrinter::newLine(char InitialChar) {
  OS << "\n";
  OS.indent(Indent) << InitialChar;
}

void DiffPrinter::printExplicit(StringRef Property, DiffResult C,
                                StringRef Left, StringRef Right) {
  newLine();
  printField(Property, DiffResult::UNSPECIFIED, AlignStyle::Right,
             PropertyWidth);
  printField(Left, C, AlignStyle::Center, FieldWidth);
  printField(Right, C, AlignStyle::Center, FieldWidth);
  printSeparatorRow();
}

void DiffPrinter::printSame(StringRef Property, StringRef Value) {
  newLine();
  printField(Property, DiffResult::UNSPECIFIED, AlignStyle::Right,
             PropertyWidth);
  printField(Value, DiffResult::IDENTICAL, AlignStyle::Center,
             FieldWidth + 1 + FieldWidth);
  printSeparatorRow();
}

void DiffPrinter::printDifferent(StringRef Property, StringRef Left,
                                 StringRef Right) {
  newLine();
  printField(Property, DiffResult::UNSPECIFIED, AlignStyle::Right,
             PropertyWidth);
  printField(Left, DiffResult::DIFFERENT, AlignStyle::Center, FieldWidth);
  printField(Right, DiffResult::DIFFERENT, AlignStyle::Center, FieldWidth);
  printSeparatorRow();
}

void DiffPrinter::printField(StringRef Value, DiffResult C, AlignStyle Style,
                             uint32_t Width) {
  if (Style == AlignStyle::Right)
    --Width;

  std::string FormattedItem =
      formatv("{0}", fmt_align(Value, Style, Width)).str();
  if (C != DiffResult::UNSPECIFIED) {
    setColor(OS, C);
    OS << FormattedItem;
    OS.resetColor();
  } else
    OS << FormattedItem;
  if (Style == AlignStyle::Right)
    OS << ' ';
  OS << '|';
}
