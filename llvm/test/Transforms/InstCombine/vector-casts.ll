; RUN: llvm-as < %s | opt -instcombine

define void @convert(<2 x i32>* %dst.addr, <2 x i64> %src) nounwind {
entry:
	%val = trunc <2 x i64> %src to <2 x i32>		; <<2 x i32>> [#uses=1]
	%add = add <2 x i32> %val, <i32 1, i32 1>		; <<2 x i32>> [#uses=1]
	store <2 x i32> %add, <2 x i32>* %dst.addr
	ret void
}

define <2 x i65> @foo(<2 x i64> %t) {
  %a = trunc <2 x i64> %t to <2 x i32>
  %b = zext <2 x i32> %a to <2 x i65>
  ret <2 x i65> %b
}
define <2 x i64> @bar(<2 x i65> %t) {
  %a = trunc <2 x i65> %t to <2 x i32>
  %b = zext <2 x i32> %a to <2 x i64>
  ret <2 x i64> %b
}
define <2 x i65> @foos(<2 x i64> %t) {
  %a = trunc <2 x i64> %t to <2 x i32>
  %b = sext <2 x i32> %a to <2 x i65>
  ret <2 x i65> %b
}
define <2 x i64> @bars(<2 x i65> %t) {
  %a = trunc <2 x i65> %t to <2 x i32>
  %b = sext <2 x i32> %a to <2 x i64>
  ret <2 x i64> %b
}
define <2 x i64> @quxs(<2 x i64> %t) {
  %a = trunc <2 x i64> %t to <2 x i32>
  %b = sext <2 x i32> %a to <2 x i64>
  ret <2 x i64> %b
}
define <2 x i64> @quxt(<2 x i64> %t) {
  %a = shl <2 x i64> %t, <i64 32, i64 32>
  %b = ashr <2 x i64> %a, <i64 32, i64 32>
  ret <2 x i64> %b
}
define <2 x double> @fa(<2 x double> %t) {
  %a = fptrunc <2 x double> %t to <2 x float>
  %b = fpext <2 x float> %a to <2 x double>
  ret <2 x double> %b
}
define <2 x double> @fb(<2 x double> %t) {
  %a = fptoui <2 x double> %t to <2 x i64>
  %b = uitofp <2 x i64> %a to <2 x double>
  ret <2 x double> %b
}
define <2 x double> @fc(<2 x double> %t) {
  %a = fptosi <2 x double> %t to <2 x i64>
  %b = sitofp <2 x i64> %a to <2 x double>
  ret <2 x double> %b
}
