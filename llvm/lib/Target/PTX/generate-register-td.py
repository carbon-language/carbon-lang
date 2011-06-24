#!/usr/bin/env python
##===- generate-register-td.py --------------------------------*-python-*--===##
##
##                     The LLVM Compiler Infrastructure
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##
##===----------------------------------------------------------------------===##
##
## This file describes the PTX register file generator.
##
##===----------------------------------------------------------------------===##

from sys import argv, exit, stdout


if len(argv) != 6:
    print('Usage: generate-register-td.py <num_preds> <num_8> <num_16> <num_32> <num_64>')
    exit(1)

try:
    num_pred  = int(argv[1])
    num_8bit  = int(argv[2])
    num_16bit = int(argv[3])
    num_32bit = int(argv[4])
    num_64bit = int(argv[5])
except:
    print('ERROR: Invalid integer parameter')
    exit(1)

## Print the register definition file
td_file = open('PTXRegisterInfo.td', 'w')

td_file.write('''
//===- PTXRegisterInfo.td - PTX Register defs ----------------*- tblgen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the PTX register file
//===----------------------------------------------------------------------===//

class PTXReg<string n> : Register<n> {
  let Namespace = "PTX";
}

//===----------------------------------------------------------------------===//
//  Registers
//===----------------------------------------------------------------------===//
''')


# Print predicate registers
td_file.write('\n///===- Predicate Registers -----------------------------------------------===//\n\n')
for r in range(0, num_pred):
    td_file.write('def P%d : PTXReg<"p%d">;\n' % (r, r))

# Print 8-bit registers
td_file.write('\n///===- 8-Bit Registers --------------------------------------------------===//\n\n')
for r in range(0, num_8bit):
    td_file.write('def RQ%d : PTXReg<"rq%d">;\n' % (r, r))

# Print 16-bit registers
td_file.write('\n///===- 16-Bit Registers --------------------------------------------------===//\n\n')
for r in range(0, num_16bit):
    td_file.write('def RH%d : PTXReg<"rh%d">;\n' % (r, r))

# Print 32-bit registers
td_file.write('\n///===- 32-Bit Registers --------------------------------------------------===//\n\n')
for r in range(0, num_32bit):
    td_file.write('def R%d : PTXReg<"r%d">;\n' % (r, r))

# Print 64-bit registers
td_file.write('\n///===- 64-Bit Registers --------------------------------------------------===//\n\n')
for r in range(0, num_64bit):
    td_file.write('def RD%d : PTXReg<"rd%d">;\n' % (r, r))


td_file.write('''
//===----------------------------------------------------------------------===//
//  Register classes
//===----------------------------------------------------------------------===//
''')


# Print register classes

td_file.write('def RegPred : RegisterClass<"PTX", [i1], 8, (sequence "P%%u", 0, %d)>;\n' % (num_pred-1))
td_file.write('def RegI8  : RegisterClass<"PTX", [i8],  8, (sequence "RQ%%u", 0, %d)>;\n' % (num_8bit-1))
td_file.write('def RegI16 : RegisterClass<"PTX", [i16], 16, (sequence "RH%%u", 0, %d)>;\n' % (num_16bit-1))
td_file.write('def RegI32 : RegisterClass<"PTX", [i32], 32, (sequence "R%%u", 0, %d)>;\n' % (num_32bit-1))
td_file.write('def RegI64 : RegisterClass<"PTX", [i64], 64, (sequence "RD%%u", 0, %d)>;\n' % (num_64bit-1))
td_file.write('def RegF32 : RegisterClass<"PTX", [f32], 32, (sequence "R%%u", 0, %d)>;\n' % (num_32bit-1))
td_file.write('def RegF64 : RegisterClass<"PTX", [f64], 64, (sequence "RD%%u", 0, %d)>;\n' % (num_64bit-1))


td_file.close()

## Now write the PTXCallingConv.td file
td_file = open('PTXCallingConv.td', 'w')

# Reserve 10% of the available registers for return values, and the other 90%
# for parameters
num_ret_pred    = int(0.1 * num_pred)
num_ret_8bit    = int(0.1 * num_8bit)
num_ret_16bit   = int(0.1 * num_16bit)
num_ret_32bit   = int(0.1 * num_32bit)
num_ret_64bit   = int(0.1 * num_64bit)
num_param_pred  = num_pred - num_ret_pred
num_param_8bit = num_8bit - num_ret_8bit
num_param_16bit = num_16bit - num_ret_16bit
num_param_32bit = num_32bit - num_ret_32bit
num_param_64bit = num_64bit - num_ret_64bit

param_regs_pred  = [('P%d' % (i+num_ret_pred)) for i in range(0, num_param_pred)]
ret_regs_pred    = ['P%d' % i for i in range(0, num_ret_pred)]
param_regs_8bit  = [('RQ%d' % (i+num_ret_8bit)) for i in range(0, num_param_8bit)]
ret_regs_8bit    = ['RQ%d' % i for i in range(0, num_ret_8bit)]
param_regs_16bit = [('RH%d' % (i+num_ret_16bit)) for i in range(0, num_param_16bit)]
ret_regs_16bit   = ['RH%d' % i for i in range(0, num_ret_16bit)]
param_regs_32bit = [('R%d' % (i+num_ret_32bit)) for i in range(0, num_param_32bit)]
ret_regs_32bit   = ['R%d' % i for i in range(0, num_ret_32bit)]
param_regs_64bit = [('RD%d' % (i+num_ret_64bit)) for i in range(0, num_param_64bit)]
ret_regs_64bit   = ['RD%d' % i for i in range(0, num_ret_64bit)]

param_list_pred  = reduce(lambda x, y: '%s, %s' % (x, y), param_regs_pred)
ret_list_pred    = reduce(lambda x, y: '%s, %s' % (x, y), ret_regs_pred)
param_list_8bit  = reduce(lambda x, y: '%s, %s' % (x, y), param_regs_8bit)
ret_list_8bit    = reduce(lambda x, y: '%s, %s' % (x, y), ret_regs_8bit)
param_list_16bit = reduce(lambda x, y: '%s, %s' % (x, y), param_regs_16bit)
ret_list_16bit   = reduce(lambda x, y: '%s, %s' % (x, y), ret_regs_16bit)
param_list_32bit = reduce(lambda x, y: '%s, %s' % (x, y), param_regs_32bit)
ret_list_32bit   = reduce(lambda x, y: '%s, %s' % (x, y), ret_regs_32bit)
param_list_64bit = reduce(lambda x, y: '%s, %s' % (x, y), param_regs_64bit)
ret_list_64bit   = reduce(lambda x, y: '%s, %s' % (x, y), ret_regs_64bit)

td_file.write('''
//===--- PTXCallingConv.td - Calling Conventions -----------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This describes the calling conventions for the PTX architecture.
//
//===----------------------------------------------------------------------===//

// PTX Formal Parameter Calling Convention
def CC_PTX : CallingConv<[
  CCIfType<[i1],      CCAssignToReg<[%s]>>,
  CCIfType<[i8],      CCAssignToReg<[%s]>>,
  CCIfType<[i16],     CCAssignToReg<[%s]>>,
  CCIfType<[i32,f32], CCAssignToReg<[%s]>>,
  CCIfType<[i64,f64], CCAssignToReg<[%s]>>
]>;

// PTX Return Value Calling Convention
def RetCC_PTX : CallingConv<[
  CCIfType<[i1],      CCAssignToReg<[%s]>>,
  CCIfType<[i8],      CCAssignToReg<[%s]>>,
  CCIfType<[i16],     CCAssignToReg<[%s]>>,
  CCIfType<[i32,f32], CCAssignToReg<[%s]>>,
  CCIfType<[i64,f64], CCAssignToReg<[%s]>>
]>;
''' % (param_list_pred, param_list_8bit, param_list_16bit, param_list_32bit, param_list_64bit,
       ret_list_pred, ret_list_8bit, ret_list_16bit, ret_list_32bit, ret_list_64bit))


td_file.close()
