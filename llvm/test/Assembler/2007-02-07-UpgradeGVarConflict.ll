; For PR1187
; RUN: llvm-upgrade < %s > /dev/null

%struct.isc_hash_t = type { uint, sbyte*, int, uint, uint, 
                            [4 x ubyte], ulong, ushort* }
%hash = internal global %struct.isc_hash_t* null

implementation

void %somefunc() {
  %key_addr = alloca sbyte*
  %tmp21 = load sbyte** %key_addr
  %tmp22 = call fastcc uint %hash(sbyte* %tmp21, uint 0)
  ret void
}

internal fastcc uint %hash(sbyte* %key, uint %case_sensitive) {
  ret uint 0
}
