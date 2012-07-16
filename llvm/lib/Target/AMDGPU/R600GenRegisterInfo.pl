#===-- R600GenRegisterInfo.pl - Script for generating register info files --===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#
#
# This perl script prints to stdout .td code to be used as R600RegisterInfo.td
# it also generates a file called R600HwRegInfo.include, which contains helper
# functions for determining the hw encoding of registers.
#
#===------------------------------------------------------------------------===#

use strict;
use warnings;

use constant CONST_REG_COUNT => 100;
use constant TEMP_REG_COUNT => 128;

my $CREG_MAX = CONST_REG_COUNT - 1;
my $TREG_MAX = TEMP_REG_COUNT - 1;

print <<STRING;

class R600Reg <string name, bits<16> encoding> : Register<name> {
  let Namespace = "AMDGPU";
  let HWEncoding = encoding;
}

class R600Reg_128<string n, list<Register> subregs, bits<16> encoding> :
    RegisterWithSubRegs<n, subregs> {
  let Namespace = "AMDGPU";
  let SubRegIndices = [sel_x, sel_y, sel_z, sel_w];
  let HWEncoding = encoding;
}

STRING

my $i;

### REG DEFS ###

my @creg_list = print_reg_defs(CONST_REG_COUNT * 4, "C");
my @treg_list = print_reg_defs(TEMP_REG_COUNT * 4, "T");

my @t128reg;
my @treg_x;
for (my $i = 0; $i < TEMP_REG_COUNT; $i++) {
  my $name = "T$i\_XYZW";
  print qq{def $name : R600Reg_128 <"T$i.XYZW", [T$i\_X, T$i\_Y, T$i\_Z, T$i\_W], $i >;\n};
  $t128reg[$i] = $name;
  $treg_x[$i] = "T$i\_X";
  if ($i % 10 == 0) {
    $t128reg[$i] .= "\n";
    $treg_x[$i] .= "\n";
  }
}

my $treg_string = join(",", @treg_list);
my $creg_list = join(",", @creg_list);
my $t128_string = join(",", @t128reg);
my $treg_x_string = join(",", @treg_x);
print <<STRING;

class RegSet <dag s> {
  dag set = s;
}

def ZERO : R600Reg<"0.0", 248>;
def ONE : R600Reg<"1.0", 249>;
def NEG_ONE : R600Reg<"-1.0", 249>;
def ONE_INT : R600Reg<"1", 250>;
def HALF : R600Reg<"0.5", 252>;
def NEG_HALF : R600Reg<"-0.5", 252>;
def PV_X : R600Reg<"pv.x", 254>;
def ALU_LITERAL_X : R600Reg<"literal.x", 253>;

def R600_CReg32 : RegisterClass <"AMDGPU", [f32, i32], 32, (add
    $creg_list)>;

def R600_TReg32 : RegisterClass <"AMDGPU", [f32, i32], 32, (add
    $treg_string)>;

def R600_TReg32_X : RegisterClass <"AMDGPU", [f32, i32], 32, (add
    $treg_x_string)>;
    
def R600_Reg32 : RegisterClass <"AMDGPU", [f32, i32], 32, (add
    R600_TReg32,
    R600_CReg32,
    ZERO, HALF, ONE, ONE_INT, PV_X, ALU_LITERAL_X, NEG_ONE, NEG_HALF)>;

def R600_Reg128 : RegisterClass<"AMDGPU", [v4f32, v4i32], 128, (add
    $t128_string)>
{
  let CopyCost = -1;
}

STRING

my %index_map;
my %chan_map;

for ($i = 0; $i <= $#creg_list; $i++) {
  push(@{$index_map{get_hw_index($i)}}, $creg_list[$i]);
  push(@{$chan_map{get_chan_str($i)}}, $creg_list[$i]);
}

for ($i = 0; $i <= $#treg_list; $i++) {
  push(@{$index_map{get_hw_index($i)}}, $treg_list[$i]);
  push(@{$chan_map{get_chan_str($i)}}, $treg_list[$i]);
}

for ($i = 0; $i <= $#t128reg; $i++) {
  push(@{$index_map{$i}}, $t128reg[$i]);
  push(@{$chan_map{'X'}}, $t128reg[$i]);
}

open(OUTFILE, ">", "R600HwRegInfo.include");

print OUTFILE <<STRING;

unsigned R600RegisterInfo::getHWRegChanGen(unsigned reg) const
{
  switch(reg) {
  default: assert(!"Unknown register"); return 0;
STRING

foreach my $key (keys(%chan_map)) {
  foreach my $reg (@{$chan_map{$key}}) {
    chomp($reg);
    print OUTFILE " case AMDGPU::$reg:\n";
  }
  my $val;
  if ($key eq 'X') {
    $val = 0;
  } elsif ($key eq 'Y') {
    $val = 1;
  } elsif ($key eq 'Z') {
    $val = 2;
  } elsif ($key eq 'W') {
    $val = 3;
  } else {
    die("Unknown chan value; $key");
  }
  print OUTFILE "    return $val;\n\n";
}

print OUTFILE "  }\n}\n\n";

sub print_reg_defs {
  my ($count, $prefix) = @_;

  my @reg_list;

  for ($i = 0; $i < $count; $i++) {
    my $hw_index = get_hw_index($i);
    my $chan= get_chan_str($i);
    my $name = "$prefix$hw_index\_$chan";
    print qq{def $name : R600Reg <"$prefix$hw_index.$chan", $hw_index>;\n};
    $reg_list[$i] = $name;
    if ($i % 10 == 0) {
        $reg_list[$i] .= "\n";
    }
  }
  return @reg_list;
}

#Helper functions
sub get_hw_index {
  my ($index) = @_;
  return int($index / 4);
}

sub get_chan_str {
  my ($index) = @_;
  my $chan = $index % 4;
  if ($chan == 0 )  {
    return 'X';
  } elsif ($chan == 1) {
    return 'Y';
  } elsif ($chan == 2) {
    return 'Z';
  } elsif ($chan == 3) {
    return 'W';
  } else {
    die("Unknown chan value: $chan");
  }
}
