// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// MakeFunc s390 implementation.

package reflect

import "unsafe"

// Convenience types and constants.
const s390_arch_stack_slot_align uintptr = 4
const s390_num_gr = 5
const s390_num_fr = 2

type s390_arch_gr_t uint32
type s390_arch_fr_t uint64

// The assembler stub will pass a pointer to this structure.
// This will come in holding all the registers that might hold
// function parameters.  On return we will set the registers that
// might hold result values.
type s390_regs struct {
	r2         s390_arch_gr_t
	r3         s390_arch_gr_t
	r4         s390_arch_gr_t
	r5         s390_arch_gr_t
	r6         s390_arch_gr_t
	stack_args s390_arch_gr_t
	f0         s390_arch_fr_t
	f2         s390_arch_fr_t
}

// Argument classifications that arise for Go types.
type s390_arg_t int

const (
	s390_general_reg s390_arg_t = iota
	s390_general_reg_pair
	s390_float_reg
	// Argument passed as a pointer to an in-memory value.
	s390_mem_ptr
	s390_empty
)

// s390ClassifyParameter returns the register class needed to
// pass the value of type TYP.  s390_empty means the register is
// not used.  The second and third return values are the offset of
// an rtype parameter passed in a register (second) or stack slot
// (third).
func s390ClassifyParameter(typ *rtype) (s390_arg_t, uintptr, uintptr) {
	offset := s390_arch_stack_slot_align - typ.Size()
	if typ.Size() > s390_arch_stack_slot_align {
		offset = 0
	}
	switch typ.Kind() {
	default:
		panic("internal error--unknown kind in s390ClassifyParameter")
	case Bool, Int, Int8, Int16, Int32, Uint, Uint8, Uint16, Uint32:
		return s390_general_reg, offset, offset
	case Int64, Uint64:
		return s390_general_reg_pair, 0, 0
	case Uintptr, Chan, Func, Map, Ptr, UnsafePointer:
		return s390_general_reg, 0, 0
	case Float32, Float64:
		return s390_float_reg, 0, offset
	case Complex64, Complex128:
		// Complex numbers are passed by reference.
		return s390_mem_ptr, 0, 0
	case Array, Struct:
		var ityp *rtype
		var length int

		if typ.Size() == 0 {
			return s390_empty, 0, 0
		}
		switch typ.Size() {
		default:
			// Pointer to memory.
			return s390_mem_ptr, 0, 0
		case 1, 2:
			// Pass in an integer register.
			return s390_general_reg, offset, offset

		case 4, 8:
			// See below.
		}
		if typ.Kind() == Array {
			atyp := (*arrayType)(unsafe.Pointer(typ))
			length = atyp.Len()
			ityp = atyp.elem
		} else {
			styp := (*structType)(unsafe.Pointer(typ))
			length = len(styp.fields)
			ityp = styp.fields[0].typ
		}
		if length == 1 {
			class, off_reg, off_slot := s390ClassifyParameter(ityp)
			if class == s390_float_reg {
				// The array (stored in a structure) or struct
				// is "equivalent to a floating point type" as
				// defined in the S390 Abi.  Note that this
				// can only be the case in the case 4 of the
				// switch above.
				return s390_float_reg, off_reg, off_slot
			}
		}
		switch typ.Size() {
		case 4:
			return s390_general_reg, offset, offset
		case 8:
			return s390_general_reg_pair, 0, 0
		default:
			return s390_general_reg, 0, 0
		}
	case Interface, String:
		// Structure of size 8.
		return s390_general_reg_pair, 0, 0

	case Slice:
		return s390_mem_ptr, 0, 0
	}
}

// s390ClassifyReturn returns the register classes needed to
// return the value of type TYP.  s390_empty means the register is
// not used.  The second value is the offset of an rtype return
// parameter if stored in a register.
func s390ClassifyReturn(typ *rtype) (s390_arg_t, uintptr) {
	offset := s390_arch_stack_slot_align - typ.Size()
	if typ.Size() > s390_arch_stack_slot_align {
		offset = 0
	}
	switch typ.Kind() {
	default:
		panic("internal error--unknown kind in s390ClassifyReturn")
	case Bool, Int, Int8, Int16, Int32,
		Uint, Uint8, Uint16, Uint32, Uintptr:

		return s390_general_reg, offset
	case Int64, Uint64:
		return s390_general_reg_pair, 0
	case Chan, Func, Map, Ptr, UnsafePointer:
		return s390_general_reg, 0
	case Float32, Float64:
		return s390_float_reg, 0
	case Complex64, Complex128:
		return s390_mem_ptr, 0
	case Interface, Slice, String:
		return s390_mem_ptr, 0
	case Array, Struct:
		if typ.size == 0 {
			return s390_empty, 0
		}
		// No optimization is done for returned structures and arrays.
		return s390_mem_ptr, 0
	}
}

// Given a value of type *rtype left aligned in an unsafe.Pointer,
// reload the value so that it can be stored in a general or
// floating point register.  For general registers the value is
// sign extend and right aligned.
func s390ReloadForRegister(typ *rtype, w uintptr, offset uintptr) uintptr {
	var do_sign_extend bool = false
	var gr s390_arch_gr_t

	switch typ.Kind() {
	case Int, Int8, Int16, Int32:
		do_sign_extend = true
	default:
		// Handle all other cases in the next switch.
	}
	switch typ.size {
	case 1:
		if do_sign_extend == true {
			se := int32(*(*int8)(unsafe.Pointer(&w)))
			gr = *(*s390_arch_gr_t)(unsafe.Pointer(&se))
		} else {
			e := int32(*(*uint8)(unsafe.Pointer(&w)))
			gr = *(*s390_arch_gr_t)(unsafe.Pointer(&e))
		}
	case 2:
		if do_sign_extend == true {
			se := int32(*(*int16)(unsafe.Pointer(&w)))
			gr = *(*s390_arch_gr_t)(unsafe.Pointer(&se))
		} else {
			e := int32(*(*uint16)(unsafe.Pointer(&w)))
			gr = *(*s390_arch_gr_t)(unsafe.Pointer(&e))
		}
	default:
		panic("reflect: bad size in s390ReloadForRegister")
	}

	return *(*uintptr)(unsafe.Pointer(&gr))
}

// MakeFuncStubGo implements the s390 calling convention for
// MakeFunc.  This should not be called.  It is exported so that
// assembly code can call it.
func S390MakeFuncStubGo(regs *s390_regs, c *makeFuncImpl) {
	ftyp := c.typ
	gr := 0
	fr := 0
	ap := uintptr(regs.stack_args)

	// See if the result requires a struct.  If it does, the first
	// parameter is a pointer to the struct.
	var ret_class s390_arg_t
	var ret_off_reg uintptr
	var ret_type *rtype

	switch len(ftyp.out) {
	case 0:
		ret_type = nil
		ret_class, ret_off_reg = s390_empty, 0
	case 1:
		ret_type = ftyp.out[0]
		ret_class, ret_off_reg = s390ClassifyReturn(ret_type)
	default:
		ret_type = nil
		ret_class, ret_off_reg = s390_mem_ptr, 0
	}
	in := make([]Value, 0, len(ftyp.in))
	if ret_class == s390_mem_ptr {
		// We are returning a value in memory, which means
		// that the first argument is a hidden parameter
		// pointing to that return area.
		gr++
	}

argloop:
	for _, rt := range ftyp.in {
		class, off_reg, off_slot := s390ClassifyParameter(rt)
		fl := flag(rt.Kind()) << flagKindShift
		switch class {
		case s390_empty:
			v := Value{rt, nil, fl | flagIndir}
			in = append(in, v)
			continue argloop
		case s390_general_reg:
			// Values stored in a general register are right
			// aligned.
			if gr < s390_num_gr {
				val := s390_general_reg_val(regs, gr)
				iw := unsafe.Pointer(&val)
				k := rt.Kind()
				if k != Ptr && k != UnsafePointer {
					ix := uintptr(unsafe.Pointer(&val))
					ix += off_reg
					iw = unsafe.Pointer(ix)
					fl |= flagIndir
				}
				v := Value{rt, iw, fl}
				in = append(in, v)
				gr++
			} else {
				in, ap = s390_add_stackreg(
					in, ap, rt, off_slot)
			}
			continue argloop
		case s390_general_reg_pair:
			// 64-bit integers and structs are passed in a register
			// pair.
			if gr+1 < s390_num_gr {
				val := uint64(s390_general_reg_val(regs, gr))<<32 + uint64(s390_general_reg_val(regs, gr+1))
				iw := unsafe.Pointer(&val)
				v := Value{rt, iw, fl | flagIndir}
				in = append(in, v)
				gr += 2
			} else {
				in, ap = s390_add_stackreg(in, ap, rt, off_slot)
				gr = s390_num_gr
			}
			continue argloop
		case s390_float_reg:
			// In a register, floats are left aligned, but in a
			// stack slot they are right aligned.
			if fr < s390_num_fr {
				val := s390_float_reg_val(regs, fr)
				ix := uintptr(unsafe.Pointer(&val))
				v := Value{
					rt, unsafe.Pointer(unsafe.Pointer(ix)),
					fl | flagIndir,
				}
				in = append(in, v)
				fr++
			} else {
				in, ap = s390_add_stackreg(
					in, ap, rt, off_slot)
			}
			continue argloop
		case s390_mem_ptr:
			if gr < s390_num_gr {
				// Register holding a pointer to memory.
				val := s390_general_reg_val(regs, gr)
				v := Value{
					rt, unsafe.Pointer(uintptr(val)),
					fl | flagIndir}
				in = append(in, v)
				gr++
			} else {
				// Stack slot holding a pointer to memory.
				in, ap = s390_add_memarg(in, ap, rt)
			}
			continue argloop
		}
		panic("reflect: argtype not handled in MakeFunc:argloop")
	}

	// All the real arguments have been found and turned into
	// Values.  Call the real function.

	out := c.call(in)

	if len(out) != len(ftyp.out) {
		panic("reflect: wrong return count from function created by MakeFunc")
	}

	for i, typ := range ftyp.out {
		v := out[i]
		if v.typ != typ {
			panic(
				"reflect: function created by MakeFunc using " +
					funcName(c.fn) + " returned wrong type: have " +
					out[i].typ.String() + " for " + typ.String())
		}
		if v.flag&flagRO != 0 {
			panic(
				"reflect: function created by MakeFunc using " +
					funcName(c.fn) + " returned value obtained " +
					"from unexported field")
		}
	}

	switch ret_class {
	case s390_general_reg, s390_float_reg, s390_general_reg_pair:
		// Single return value in a general or floating point register.
		v := out[0]
		var w uintptr
		if v.Kind() == Ptr || v.Kind() == UnsafePointer {
			w = uintptr(v.pointer())
		} else {
			w = uintptr(loadScalar(v.ptr, v.typ.size))
			if ret_off_reg != 0 {
				w = s390ReloadForRegister(
					ret_type, w, ret_off_reg)
			}
		}
		if ret_class == s390_float_reg {
			regs.f0 = s390_arch_fr_t(uintptr(w))
		} else if ret_class == s390_general_reg {
			regs.r2 = s390_arch_gr_t(uintptr(w))
		} else {
			regs.r2 = s390_arch_gr_t(uintptr(w) >> 32)
			regs.r3 = s390_arch_gr_t(uintptr(w) & 0xffffffff)
		}

	case s390_mem_ptr:
		// The address of the memory area was passed as a hidden
		// parameter in %r2.  Multiple return values are always returned
		// in an in-memory structure.
		ptr := unsafe.Pointer(uintptr(regs.r2))
		off := uintptr(0)
		for i, typ := range ftyp.out {
			v := out[i]
			off = align(off, uintptr(typ.fieldAlign))
			addr := unsafe.Pointer(uintptr(ptr) + off)
			if v.flag&flagIndir == 0 && (v.kind() == Ptr || v.kind() == UnsafePointer) {
				*(*unsafe.Pointer)(addr) = v.ptr
			} else {
				memmove(addr, v.ptr, typ.size)
			}
			off += typ.size
		}

	case s390_empty:
	}

	return
}

// The s390_add_stackreg function adds an argument passed on the
// stack that could be passed in a register.
func s390_add_stackreg(in []Value, ap uintptr, rt *rtype, offset uintptr) ([]Value, uintptr) {
	// If we're not already at the beginning of a stack slot, round up to
	// the beginning of the next one.
	ap = align(ap, s390_arch_stack_slot_align)
	// If offset is > 0, the data is right aligned on the stack slot.
	ap += offset

	// We have to copy the argument onto the heap in case the
	// function hangs onto the reflect.Value we pass it.
	p := unsafe_New(rt)
	memmove(p, unsafe.Pointer(ap), rt.size)

	v := Value{rt, p, flag(rt.Kind()<<flagKindShift) | flagIndir}
	in = append(in, v)
	ap += rt.size
	ap = align(ap, s390_arch_stack_slot_align)

	return in, ap
}

// The s390_add_memarg function adds an argument passed in memory.
func s390_add_memarg(in []Value, ap uintptr, rt *rtype) ([]Value, uintptr) {
	// If we're not already at the beginning of a stack slot,
	// round up to the beginning of the next one.
	ap = align(ap, s390_arch_stack_slot_align)

	// We have to copy the argument onto the heap in case the
	// function hangs onto the reflect.Value we pass it.
	p := unsafe_New(rt)
	memmove(p, *(*unsafe.Pointer)(unsafe.Pointer(ap)), rt.size)

	v := Value{rt, p, flag(rt.Kind()<<flagKindShift) | flagIndir}
	in = append(in, v)
	ap += s390_arch_stack_slot_align

	return in, ap
}

// The s390_general_reg_val function returns the value of integer register GR.
func s390_general_reg_val(regs *s390_regs, gr int) s390_arch_gr_t {
	switch gr {
	case 0:
		return regs.r2
	case 1:
		return regs.r3
	case 2:
		return regs.r4
	case 3:
		return regs.r5
	case 4:
		return regs.r6
	default:
		panic("s390_general_reg_val: bad integer register")
	}
}

// The s390_float_reg_val function returns the value of float register FR.
func s390_float_reg_val(regs *s390_regs, fr int) uintptr {
	var r s390_arch_fr_t
	switch fr {
	case 0:
		r = regs.f0
	case 1:
		r = regs.f2
	default:
		panic("s390_float_reg_val: bad floating point register")
	}
	return uintptr(r)
}
