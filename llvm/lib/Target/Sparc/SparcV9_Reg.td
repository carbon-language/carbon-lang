//===- Sparc.td - Target Description for Sparc V9 Target --------*- C++ -*-===//
// vim:ft=cpp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Declarations that describe the Sparc register file 
//===----------------------------------------------------------------------===//

class V9Reg : Register { set Namespace = "SparcV9"; }

// Ri - One of the 32 64 bit integer registers
class Ri<bits<5> num> : V9Reg { set Size = 64; field bits<5> Num = num; }

def G0 : Ri< 0>;    def G1 : Ri< 1>;    def G2 : Ri< 2>;    def G3 : Ri< 3>;
def G4 : Ri< 4>;    def G5 : Ri< 5>;    def G6 : Ri< 6>;    def G7 : Ri< 7>;
def O0 : Ri< 8>;    def O1 : Ri< 9>;    def O2 : Ri<10>;    def O3 : Ri<11>;
def O4 : Ri<12>;    def O5 : Ri<13>;    def O6 : Ri<14>;    def O7 : Ri<15>;
def L0 : Ri<16>;    def L1 : Ri<17>;    def L2 : Ri<18>;    def L3 : Ri<19>;
def L4 : Ri<20>;    def L5 : Ri<21>;    def L6 : Ri<22>;    def L7 : Ri<23>;
def I0 : Ri<24>;    def I1 : Ri<25>;    def I2 : Ri<26>;    def I3 : Ri<27>;
def I4 : Ri<28>;    def I5 : Ri<29>;    def I6 : Ri<30>;    def I7 : Ri<31>;
// Floating-point registers?
// ...
