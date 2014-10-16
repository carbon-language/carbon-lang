//===- ir_test.go - Tests for ir ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests bindings for the ir component.
//
//===----------------------------------------------------------------------===//

package llvm

import (
	"strings"
	"testing"
)

func testAttribute(t *testing.T, attr Attribute, name string) {
	mod := NewModule("")
	defer mod.Dispose()

	ftyp := FunctionType(VoidType(), nil, false)
	fn := AddFunction(mod, "foo", ftyp)

	fn.AddFunctionAttr(attr)
	newattr := fn.FunctionAttr()
	if attr != newattr {
		t.Errorf("got attribute mask %d, want %d", newattr, attr)
	}

	text := mod.String()
	if !strings.Contains(text, " "+name+" ") {
		t.Errorf("expected attribute '%s', got:\n%s", name, text)
	}

	fn.RemoveFunctionAttr(attr)
	newattr = fn.FunctionAttr()
	if newattr != 0 {
		t.Errorf("got attribute mask %d, want 0", newattr)
	}
}

func TestAttributes(t *testing.T) {
	// Tests that our attribute constants haven't drifted from LLVM's.
	attrTests := []struct {
		attr Attribute
		name string
	}{
		{SanitizeAddressAttribute, "sanitize_address"},
		{AlwaysInlineAttribute, "alwaysinline"},
		{BuiltinAttribute, "builtin"},
		{ByValAttribute, "byval"},
		{InAllocaAttribute, "inalloca"},
		{InlineHintAttribute, "inlinehint"},
		{InRegAttribute, "inreg"},
		{JumpTableAttribute, "jumptable"},
		{MinSizeAttribute, "minsize"},
		{NakedAttribute, "naked"},
		{NestAttribute, "nest"},
		{NoAliasAttribute, "noalias"},
		{NoBuiltinAttribute, "nobuiltin"},
		{NoCaptureAttribute, "nocapture"},
		{NoDuplicateAttribute, "noduplicate"},
		{NoImplicitFloatAttribute, "noimplicitfloat"},
		{NoInlineAttribute, "noinline"},
		{NonLazyBindAttribute, "nonlazybind"},
		{NonNullAttribute, "nonnull"},
		{NoRedZoneAttribute, "noredzone"},
		{NoReturnAttribute, "noreturn"},
		{NoUnwindAttribute, "nounwind"},
		{OptimizeNoneAttribute, "optnone"},
		{OptimizeForSizeAttribute, "optsize"},
		{ReadNoneAttribute, "readnone"},
		{ReadOnlyAttribute, "readonly"},
		{ReturnedAttribute, "returned"},
		{ReturnsTwiceAttribute, "returns_twice"},
		{SExtAttribute, "signext"},
		{StackProtectAttribute, "ssp"},
		{StackProtectReqAttribute, "sspreq"},
		{StackProtectStrongAttribute, "sspstrong"},
		{StructRetAttribute, "sret"},
		{SanitizeThreadAttribute, "sanitize_thread"},
		{SanitizeMemoryAttribute, "sanitize_memory"},
		{UWTableAttribute, "uwtable"},
		{ZExtAttribute, "zeroext"},
		{ColdAttribute, "cold"},
	}

	for _, a := range attrTests {
		testAttribute(t, a.attr, a.name)
	}
}
