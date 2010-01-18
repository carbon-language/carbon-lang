// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface G
@end

@interface F
- (void)bar:(id *)objects;
- (void)foo:(G**)objects;
@end


void a() {
	F *b;
	G **keys;
	[b bar:keys];

	id *PID;
	[b foo:PID];

}

