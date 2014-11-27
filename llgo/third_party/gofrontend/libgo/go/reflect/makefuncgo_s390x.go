// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// MakeFunc s390x implementation.

package reflect

import "unsafe"

// Convenience types and constants.
const s390x_arch_stack_slot_align uintptr = 8
const s390x_num_gr = 5
const s390x_num_fr = 4

type s390x_arch_gr_t uint64
type s390x_arch_fr_t uint64

// The assembler stub will pass a pointer to this structure.
// This will come in holding all the registers that might hold
// function parameters.  On return we will set the registers that
// might hold result values.
type s390x_regs struct {
	r2         s390x_arch_gr_t
	r3         s390x_arch_gr_t
	r4         s390x_arch_gr_t
	r5         s390x_arch_gr_t
	r6         s390x_arch_gr_t
	stack_args s390x_arch_gr_t
	f0         s390x_arch_fr_t
	f2         s390x_arch_fr_t
	f4         s390x_arch_fr_t
	f6         s390x_arch_fr_t
}

// Argument classifications that arise for Go types.
type s390x_arg_t int

const (
	s390x_general_reg s390x_arg_t = iota
	s390x_float_reg
	// Argument passed as a pointer to an in-memory value.
	s390x_mem_ptr
	s390x_empty
)

// s390xClassifyParameter returns the register class needed to
// pass the value of type TYP.  s390x_empty means the register is
// not used.  The second and third return values are the offset of
// an rtype parameter passed in a register (second) or stack slot
// (third).
func s390xClassifyParameter(typ *rtype) (s390x_arg_t, uintptr, uintptr) {
	offset := s390x_arch_stack_slot_align - typ.Size()
	switch typ.Kind() {
	default:
		panic("internal error--unknown kind in s390xClassifyParameter")
	case Bool, Int, Int8, Int16, Int32, Uint, Uint8, Uint16, Uint32:
		return s390x_general_reg, offset, offset
	case Int64, Uint64, Uintptr, Chan, Func, Map, Ptr, UnsafePointer:
		return s390x_general_reg, 0, 0
	case Float32, Float64:
		return s390x_float_reg, 0, offset
	case Complex64, Complex128:
		// Complex numbers are passed by reference.
		return s390x_mem_ptr, 0, 0
	case Array, Struct:
		var ityp *rtype
		var length int

		if typ.Size() == 0 {
			return s390x_empty, 0, 0
		}
		switch typ.Size() {
		default:
			// Pointer to memory.
			return s390x_mem_ptr, 0, 0
		case 1, 2:
			// Pass in an integer register.
			return s390x_general_reg, offset, offset

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
			class, off_reg, off_slot := s390xClassifyParameter(ityp)
			if class == s390x_float_reg {
				// The array (stored in a structure) or struct
				// is "equivalent to a floating point type" as
				// defined in the S390x Abi.  Note that this
				// can only be the case in the cases 4 and 8 of
				// the switch above.
				return s390x_float_reg, off_reg, off_slot
			}
		}
		// Otherwise pass in an integer register.
		switch typ.Size() {
		case 4, 8:
			return s390x_general_reg, offset, offset
		default:
			return s390x_general_reg, 0, 0
		}
	case Interface, Slice, String:
		return s390x_mem_ptr, 0, 0
	}
}

// s390xClassifyReturn returns the register classes needed to
// return the value of type TYP.  s390_empty means the register is
// not used.  The second value is the offset of an rtype return
// parameter if stored in a register.
func s390xClassifyReturn(typ *rtype) (s390x_arg_t, uintptr) {
	offset := s390x_arch_stack_slot_align - typ.Size()
	switch typ.Kind() {
	default:
		panic("internal error--unknown kind in s390xClassifyReturn")
	case Bool, Int, Int8, Int16, Int32, Int64,
		Uint, Uint8, Uint16, Uint32, Uint64, Uintptr:

		return s390x_general_reg, offset
	case Chan, Func, Map, Ptr, UnsafePointer:
		return s390x_general_reg, 0
	case Float32, Float64:
		return s390x_float_reg, 0
	case Complex64, Complex128:
		return s390x_mem_ptr, 0
	case Interface, Slice, String:
		return s390x_mem_ptr, 0
	case Array, Struct:
		if typ.size == 0 {
			return s390x_empty, 0
		}
		// No optimization is done for returned structures and arrays.
		return s390x_mem_ptr, 0
	}
}

// Given a value of type *rtype left aligned in an unsafe.Pointer,
// reload the value so that it can be stored in a general or
// floating point register.  For general registers the value is
// sign extend and right aligned.
func s390xReloadForRegister(typ *rtype, w uintptr, offset uintptr) uintptr {
	var do_sign_extend bool = false
	var gr s390x_arch_gr_t

	switch typ.Kind() {
	case Int, Int8, Int16, Int32, Int64:
		do_sign_extend = true
	default:
		// Handle all other cases in the next switch.
	}
	switch typ.size {
	case 1:
		if do_sign_extend == true {
			se := int64(*(*int8)(unsafe.Pointer(&w)))
			gr = *(*s390x_arch_gr_t)(unsafe.Pointer(&se))
		} else {
			e := int64(*(*uint8)(unsafe.Pointer(&w)))
			gr = *(*s390x_arch_gr_t)(unsafe.Pointer(&e))
		}
	case 2:
		if do_sign_extend == true {
			se := int64(*(*int16)(unsafe.Pointer(&w)))
			gr = *(*s390x_arch_gr_t)(unsafe.Pointer(&se))
		} else {
			e := int64(*(*uint16)(unsafe.Pointer(&w)))
			gr = *(*s390x_arch_gr_t)(unsafe.Pointer(&e))
		}
	case 4:
		if do_sign_extend == true {
			se := int64(*(*int32)(unsafe.Pointer(&w)))
			gr = *(*s390x_arch_gr_t)(unsafe.Pointer(&se))
		} else {
			e := int64(*(*uint32)(unsafe.Pointer(&w)))
			gr = *(*s390x_arch_gr_t)(unsafe.Pointer(&e))
		}
	default:
		panic("reflect: bad size in s390xReloadForRegister")
	}

	return *(*uintptr)(unsafe.Pointer(&gr))
}

// MakeFuncStubGo implements the s390x calling convention for
// MakeFunc.  This should not be called.  It is exported so that
// assembly code can call it.
func S390xMakeFuncStubGo(regs *s390x_regs, c *makeFuncImpl) {
	ftyp := c.typ
	gr := 0
	fr := 0
	ap := uintptr(regs.stack_args)

	// See if the result requires a struct.  If it does, the first
	// parameter is a pointer to the struct.
	var ret_class s390x_arg_t
	var ret_off_reg uintptr
	var ret_type *rtype

	switch len(ftyp.out) {
	case 0:
		ret_type = nil
		ret_class, ret_off_reg = s390x_empty, 0
	case 1:
		ret_type = ftyp.out[0]
		ret_class, ret_off_reg = s390xClassifyReturn(ret_type)
	default:
		ret_type = nil
		ret_class, ret_off_reg = s390x_mem_ptr, 0
	}
	in := make([]Value, 0, len(ftyp.in))
	if ret_class == s390x_mem_ptr {
		// We are returning a value in memory, which means
		// that the first argument is a hidden parameter
		// pointing to that return area.
		gr++
	}

argloop:
	for _, rt := range ftyp.in {
		class, off_reg, off_slot := s390xClassifyParameter(rt)
		fl := flag(rt.Kind()) << flagKindShift
		switch class {
		case s390x_empty:
			v := Value{rt, nil, fl | flagIndir}
			in = append(in, v)
			continue argloop
		case s390x_general_reg:
			// Values stored in a general register are right
			// aligned.
			if gr < s390x_num_gr {
				val := s390x_general_reg_val(regs, gr)
				iw := unsafe.Pointer(val)
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
				in, ap = s390x_add_stackreg(
					in, ap, rt, off_slot)
			}
			continue argloop
		case s390x_float_reg:
			// In a register, floats are left aligned, but in a
			// stack slot they are right aligned.
			if fr < s390x_num_fr {
				val := s390x_float_reg_val(regs, fr)
				ix := uintptr(unsafe.Pointer(&val))
				v := Value{
					rt, unsafe.Pointer(unsafe.Pointer(ix)),
					fl | flagIndir,
				}
				in = append(in, v)
				fr++
			} else {
				in, ap = s390x_add_stackreg(
					in, ap, rt, off_slot)
			}
			continue argloop
		case s390x_mem_ptr:
			if gr < s390x_num_gr {
				// Register holding a pointer to memory.
				val := s390x_general_reg_val(regs, gr)
				v := Value{
					rt, unsafe.Pointer(val), fl | flagIndir}
				in = append(in, v)
				gr++
			} else {
				// Stack slot holding a pointer to memory.
				in, ap = s390x_add_memarg(in, ap, rt)
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
	case s390x_general_reg, s390x_float_reg:
		// Single return value in a general or floating point register.
		v := out[0]
		var w uintptr
		if v.Kind() == Ptr || v.Kind() == UnsafePointer {
			w = uintptr(v.pointer())
		} else {
			w = uintptr(loadScalar(v.ptr, v.typ.size))
			if ret_off_reg != 0 {
				w = s390xReloadForRegister(
					ret_type, w, ret_off_reg)
			}
		}
		if ret_class == s390x_float_reg {
			regs.f0 = s390x_arch_fr_t(w)
		} else {
			regs.r2 = s390x_arch_gr_t(w)
		}

	case s390x_mem_ptr:
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

	case s390x_empty:
	}

	return
}

// The s390x_add_stackreg function adds an argument passed on the
// stack that could be passed in a register.
func s390x_add_stackreg(in []Value, ap uintptr, rt *rtype, offset uintptr) ([]Value, uintptr) {
	// If we're not already at the beginning of a stack slot, round up to
	// the beginning of the next one.
	ap = align(ap, s390x_arch_stack_slot_align)
	// If offset is > 0, the data is right aligned on the stack slot.
	ap += offset

	// We have to copy the argument onto the heap in case the
	// function hangs onto the reflect.Value we pass it.
	p := unsafe_New(rt)
	memmove(p, unsafe.Pointer(ap), rt.size)

	v := Value{rt, p, flag(rt.Kind()<<flagKindShift) | flagIndir}
	in = append(in, v)
	ap += rt.size
	ap = align(ap, s390x_arch_stack_slot_align)

	return in, ap
}

// The s390x_add_memarg function adds an argument passed in memory.
func s390x_add_memarg(in []Value, ap uintptr, rt *rtype) ([]Value, uintptr) {
	// If we're not already at the beginning of a stack slot,
	// round up to the beginning of the next one.
	ap = align(ap, s390x_arch_stack_slot_align)

	// We have to copy the argument onto the heap in case the
	// function hangs onto the reflect.Value we pass it.
	p := unsafe_New(rt)
	memmove(p, *(*unsafe.Pointer)(unsafe.Pointer(ap)), rt.size)

	v := Value{rt, p, flag(rt.Kind()<<flagKindShift) | flagIndir}
	in = append(in, v)
	ap += s390x_arch_stack_slot_align

	return in, ap
}

// The s390x_general_reg_val function returns the value of integer register GR.
func s390x_general_reg_val(regs *s390x_regs, gr int) uintptr {
	var r s390x_arch_gr_t
	switch gr {
	case 0:
		r = regs.r2
	case 1:
		r = regs.r3
	case 2:
		r = regs.r4
	case 3:
		r = regs.r5
	case 4:
		r = regs.r6
	default:
		panic("s390x_general_reg_val: bad integer register")
	}
	return uintptr(r)
}

// The s390x_float_reg_val function returns the value of float register FR.
func s390x_float_reg_val(regs *s390x_regs, fr int) uintptr {
	var r s390x_arch_fr_t
	switch fr {
	case 0:
		r = regs.f0
	case 1:
		r = regs.f2
	case 2:
		r = regs.f4
	case 3:
		r = regs.f6
	default:
		panic("s390x_float_reg_val: bad floating point register")
	}
	return uintptr(r)
}
