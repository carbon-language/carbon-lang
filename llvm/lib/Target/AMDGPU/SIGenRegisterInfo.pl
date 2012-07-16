#===-- SIGenRegisterInfo.pl - Script for generating register info files ----===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#
#
# This perl script prints to stdout .td code to be used as SIRegisterInfo.td
# it also generates a file called SIHwRegInfo.include, which contains helper
# functions for determining the hw encoding of registers.
#
#===------------------------------------------------------------------------===#

use strict;
use warnings;

my $SGPR_COUNT = 104;
my $VGPR_COUNT = 256;

my $SGPR_MAX_IDX = $SGPR_COUNT - 1;
my $VGPR_MAX_IDX = $VGPR_COUNT - 1;

my $INDEX_FILE = defined($ARGV[0]) ? $ARGV[0] : '';

print <<STRING;

let Namespace = "AMDGPU" in {
  def low : SubRegIndex;
  def high : SubRegIndex;

  def sub0 : SubRegIndex;
  def sub1 : SubRegIndex;
  def sub2 : SubRegIndex;
  def sub3 : SubRegIndex;
  def sub4 : SubRegIndex;
  def sub5 : SubRegIndex;
  def sub6 : SubRegIndex;
  def sub7 : SubRegIndex;
}

class SIReg <string n, bits<16> encoding = 0> : Register<n> {
  let Namespace = "AMDGPU";
  let HWEncoding = encoding;
}

class SI_64 <string n, list<Register> subregs, bits<16> encoding> : RegisterWithSubRegs<n, subregs> {
  let Namespace = "AMDGPU";
  let SubRegIndices = [low, high];
  let HWEncoding = encoding;
}

class SI_128 <string n, list<Register> subregs, bits<16> encoding> : RegisterWithSubRegs<n, subregs> {
  let Namespace = "AMDGPU";
  let SubRegIndices = [sel_x, sel_y, sel_z, sel_w];
  let HWEncoding = encoding;
}

class SI_256 <string n, list<Register> subregs, bits<16> encoding> : RegisterWithSubRegs<n, subregs> {
  let Namespace = "AMDGPU";
  let SubRegIndices = [sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7];
  let HWEncoding = encoding;
}

class SGPR_32 <bits<16> num, string name> : SIReg<name, num>;

class VGPR_32 <bits<16> num, string name> : SIReg<name, num>;

class SGPR_64 <bits<16> num, string name, list<Register> subregs> :
    SI_64 <name, subregs, num>;

class VGPR_64 <bits<16> num, string name, list<Register> subregs> :
    SI_64 <name, subregs, num>;

class SGPR_128 <bits<16> num, string name, list<Register> subregs> :
    SI_128 <name, subregs, num>;

class VGPR_128 <bits<16> num, string name, list<Register> subregs> :
    SI_128 <name, subregs, num>;

class SGPR_256 <bits<16> num, string name, list<Register> subregs> :
    SI_256 <name, subregs, num>;

def VCC : SIReg<"VCC">;
def SCC : SIReg<"SCC">;
def SREG_LIT_0 : SIReg <"S LIT 0", 128>;

def M0 : SIReg <"M0", 124>;

//Interpolation registers

def PERSP_SAMPLE_I : SIReg <"PERSP_SAMPLE_I">;
def PERSP_SAMPLE_J : SIReg <"PERSP_SAMPLE_J">;
def PERSP_CENTER_I : SIReg <"PERSP_CENTER_I">;
def PERSP_CENTER_J : SIReg <"PERSP_CENTER_J">;
def PERSP_CENTROID_I : SIReg <"PERSP_CENTROID_I">;
def PERSP_CENTROID_J : SIReg <"PERP_CENTROID_J">;
def PERSP_I_W : SIReg <"PERSP_I_W">;
def PERSP_J_W : SIReg <"PERSP_J_W">;
def PERSP_1_W : SIReg <"PERSP_1_W">;
def LINEAR_SAMPLE_I : SIReg <"LINEAR_SAMPLE_I">;
def LINEAR_SAMPLE_J : SIReg <"LINEAR_SAMPLE_J">;
def LINEAR_CENTER_I : SIReg <"LINEAR_CENTER_I">;
def LINEAR_CENTER_J : SIReg <"LINEAR_CENTER_J">;
def LINEAR_CENTROID_I : SIReg <"LINEAR_CENTROID_I">;
def LINEAR_CENTROID_J : SIReg <"LINEAR_CENTROID_J">;
def LINE_STIPPLE_TEX_COORD : SIReg <"LINE_STIPPLE_TEX_COORD">;
def POS_X_FLOAT : SIReg <"POS_X_FLOAT">;
def POS_Y_FLOAT : SIReg <"POS_Y_FLOAT">;
def POS_Z_FLOAT : SIReg <"POS_Z_FLOAT">;
def POS_W_FLOAT : SIReg <"POS_W_FLOAT">;
def FRONT_FACE : SIReg <"FRONT_FACE">;
def ANCILLARY : SIReg <"ANCILLARY">;
def SAMPLE_COVERAGE : SIReg <"SAMPLE_COVERAGE">;
def POS_FIXED_PT : SIReg <"POS_FIXED_PT">;

STRING

#32 bit register

my @SGPR;
for (my $i = 0; $i < $SGPR_COUNT; $i++) {
  print "def SGPR$i : SGPR_32 <$i, \"SGPR$i\">;\n";
  $SGPR[$i] = "SGPR$i";
}

my @VGPR;
for (my $i = 0; $i < $VGPR_COUNT; $i++) {
  print "def VGPR$i : VGPR_32 <$i, \"VGPR$i\">;\n";
  $VGPR[$i] = "VGPR$i";
}

print <<STRING;

def SReg_32 : RegisterClass<"AMDGPU", [f32, i32], 32,
    (add (sequence "SGPR%u", 0, $SGPR_MAX_IDX),  SREG_LIT_0, M0)
>;

def VReg_32 : RegisterClass<"AMDGPU", [f32, i32], 32,
    (add (sequence "VGPR%u", 0, $VGPR_MAX_IDX),
    PERSP_SAMPLE_I, PERSP_SAMPLE_J,
    PERSP_CENTER_I, PERSP_CENTER_J,
    PERSP_CENTROID_I, PERSP_CENTROID_J,
    PERSP_I_W, PERSP_J_W, PERSP_1_W,
    LINEAR_SAMPLE_I, LINEAR_SAMPLE_J,
    LINEAR_CENTER_I, LINEAR_CENTER_J,
    LINEAR_CENTROID_I, LINEAR_CENTROID_J,
    LINE_STIPPLE_TEX_COORD,
    POS_X_FLOAT,
    POS_Y_FLOAT,
    POS_Z_FLOAT,
    POS_W_FLOAT,
    FRONT_FACE,
    ANCILLARY,
    SAMPLE_COVERAGE,
    POS_FIXED_PT
    )
>;

def AllReg_32 : RegisterClass<"AMDGPU", [f32, i32], 32,
    (add VReg_32, SReg_32) 
>;

def CCReg : RegisterClass<"AMDGPU", [f32], 32, (add VCC, SCC)>;

STRING

my @subregs_64 = ('low', 'high');
my @subregs_128 = ('sel_x', 'sel_y', 'sel_z', 'sel_w');
my @subregs_256 = ('sub0', 'sub1', 'sub2', 'sub3', 'sub4', 'sub5', 'sub6', 'sub7');

my @SGPR64 = print_sgpr_class(64, \@subregs_64, ('i64'));
my @SGPR128 = print_sgpr_class(128, \@subregs_128, ('v4f32', 'v4i32'));
my @SGPR256 = print_sgpr_class(256, \@subregs_256, ('v8i32'));

my @VGPR64 = print_vgpr_class(64, \@subregs_64, ('i64'));
my @VGPR128 = print_vgpr_class(128, \@subregs_128, ('v4f32'));


my $sgpr64_list = join(',', @SGPR64);
my $vgpr64_list = join(',', @VGPR64);
print <<STRING;

def AllReg_64 : RegisterClass<"AMDGPU", [f64, i64], 64,
    (add $sgpr64_list, $vgpr64_list)
>;

STRING

sub print_sgpr_class {
  my ($reg_width, $sub_reg_ref, @types) = @_;
  return print_reg_class('SReg', 'SGPR', $reg_width, $SGPR_COUNT, $sub_reg_ref, @types);
}

sub print_vgpr_class {
  my ($reg_width, $sub_reg_ref, @types) = @_;
  return print_reg_class('VReg', 'VGPR', $reg_width, $VGPR_COUNT, $sub_reg_ref, @types);
}

sub print_reg_class {
  my ($class_prefix, $reg_prefix, $reg_width, $reg_count, $sub_reg_ref, @types) = @_;
  my @registers;
  my $component_count = $reg_width / 32;

  for (my $i = 0; $i < $reg_count; $i += $component_count) {
    my $reg_name = $reg_prefix . $i . '_' . $reg_width;
    my @sub_regs;
    for (my $idx = 0; $idx < $component_count; $idx++) {
      my $sub_idx = $i + $idx;
      push(@sub_regs, $reg_prefix . $sub_idx);
    }
    print "def $reg_name : $reg_prefix\_$reg_width <$i, \"$reg_name\", [ ", join(',', @sub_regs) , "]>;\n";
    if ($i % 10 == 0) {
      $reg_name .= "\n";
    }
    push (@registers, $reg_name);
  }
  my $reg_list = join(', ', @registers);

  print "def $class_prefix\_$reg_width : RegisterClass<\"AMDGPU\", [" . join (', ', @types) . "], $reg_width,\n  (add $reg_list)\n>{\n";
  print "}\n";
  return @registers;
}
