//===-- X86Implicit.cpp - All the implicit uses and defs for X86 ops ------===//
//
// This defines a class which maps X86 opcodes to the registers that they
// implicitly modify or use.
//
//===----------------------------------------------------------------------===//

#include "X86Implicit.h"
#include <map>
#include <vector>

X86Implicit::X86Implicit ()
{
	implicitUses[X86::CBW].push_back (X86::AL);
	implicitDefs[X86::CBW].push_back (X86::AX);

	implicitUses[X86::CDQ].push_back (X86::EAX);
	implicitDefs[X86::CDQ].push_back (X86::EDX);

	implicitUses[X86::CWD].push_back (X86::AX);
	implicitDefs[X86::CWD].push_back (X86::DX);

	implicitUses[X86::DIVrr16].push_back (X86::DX);
	implicitUses[X86::DIVrr16].push_back (X86::AX);
	implicitDefs[X86::DIVrr16].push_back (X86::DX);
	implicitDefs[X86::DIVrr16].push_back (X86::AX);

	implicitUses[X86::DIVrr32].push_back (X86::EDX);
	implicitUses[X86::DIVrr32].push_back (X86::EAX);
	implicitDefs[X86::DIVrr32].push_back (X86::EDX);
	implicitDefs[X86::DIVrr32].push_back (X86::EAX);

	implicitUses[X86::DIVrr8].push_back (X86::AX);
	implicitDefs[X86::DIVrr8].push_back (X86::AL);
	implicitDefs[X86::DIVrr8].push_back (X86::AH);

	implicitDefs[X86::FNSTSWr8].push_back (X86::AX);

	implicitUses[X86::IDIVrr16].push_back (X86::DX);
	implicitUses[X86::IDIVrr16].push_back (X86::AX);
	implicitDefs[X86::IDIVrr16].push_back (X86::DX);
	implicitDefs[X86::IDIVrr16].push_back (X86::AX);

	implicitUses[X86::IDIVrr32].push_back (X86::EDX);
	implicitUses[X86::IDIVrr32].push_back (X86::EAX);
	implicitDefs[X86::IDIVrr32].push_back (X86::EDX);
	implicitDefs[X86::IDIVrr32].push_back (X86::EAX);

	implicitUses[X86::IDIVrr8].push_back (X86::AX);
	implicitDefs[X86::IDIVrr8].push_back (X86::AL);
	implicitDefs[X86::IDIVrr8].push_back (X86::AH);

	implicitUses[X86::LEAVE].push_back (X86::EBP);
	implicitDefs[X86::LEAVE].push_back (X86::EBP);

	implicitUses[X86::MULrr16].push_back (X86::AX);
	implicitDefs[X86::MULrr16].push_back (X86::DX);
	implicitDefs[X86::MULrr16].push_back (X86::AX);

	implicitUses[X86::MULrr32].push_back (X86::EAX);
	implicitDefs[X86::MULrr32].push_back (X86::EDX);
	implicitDefs[X86::MULrr32].push_back (X86::EAX);

	implicitUses[X86::MULrr8].push_back (X86::AL);
	implicitDefs[X86::MULrr8].push_back (X86::AX);

	implicitUses[X86::SAHF].push_back (X86::AH);

	implicitUses[X86::SARrr16].push_back (X86::CL);

	implicitUses[X86::SARrr32].push_back (X86::CL);

	implicitUses[X86::SARrr8].push_back (X86::CL);

	implicitUses[X86::SHLrr16].push_back (X86::CL);

	implicitUses[X86::SHLrr32].push_back (X86::CL);

	implicitUses[X86::SHLrr8].push_back (X86::CL);

	implicitUses[X86::SHRrr16].push_back (X86::CL);

	implicitUses[X86::SHRrr32].push_back (X86::CL);

	implicitUses[X86::SHRrr8].push_back (X86::CL);
}

