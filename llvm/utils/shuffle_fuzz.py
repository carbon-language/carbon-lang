#!/usr/bin/env python

"""A shuffle vector fuzz tester.

This is a python program to fuzz test the LLVM shufflevector instruction. It
generates a function with a random sequnece of shufflevectors, maintaining the
element mapping accumulated across the function. It then generates a main
function which calls it with a different value in each element and checks that
the result matches the expected mapping.

Take the output IR printed to stdout, compile it to an executable using whatever
set of transforms you want to test, and run the program. If it crashes, it found
a bug.
"""

import argparse
import itertools
import random
import sys

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('seed',
                      help='A string used to seed the RNG')
  parser.add_argument('-v', '--verbose', action='store_true',
                      help='Show verbose output')
  parser.add_argument('--fixed-num-shuffles', type=int,
                      help='Specify a fixed number of shuffles to test')
  parser.add_argument('--fixed-bit-width', type=int, choices=[128, 256],
                      help='Specify a fixed bit width of vector to test')
  parser.add_argument('--triple',
                      help='Specify a triple string to include in the IR')
  args = parser.parse_args()

  random.seed(args.seed)

  if args.fixed_bit_width is not None:
    if args.fixed_bit_width == 128:
      (width, element_type) = random.choice(
          [(2, 'i64'), (4, 'i32'), (8, 'i16'), (16, 'i8'),
           (2, 'f64'), (4, 'f32')])
    elif args.fixed_bit_width == 256:
      (width, element_type) = random.choice([
          (4, 'i64'), (8, 'i32'), (16, 'i16'), (32, 'i8'),
          (4, 'f64'), (8, 'f32')])
    else:
      sys.exit(1) # Checked above by argument parsing.
  else:
    width = random.choice([2, 4, 8, 16, 32, 64])
    element_type = random.choice(['i8', 'i16', 'i32', 'i64', 'f32', 'f64'])

  # FIXME: Support blends.
  shuffle_indices = [-1] + range(width)

  if args.fixed_num_shuffles is not None:
    num_shuffles = args.fixed_num_shuffles
  else:
    num_shuffles = random.randint(0, 16)

  shuffles = [[random.choice(shuffle_indices)
               for _ in itertools.repeat(None, width)]
              for _ in itertools.repeat(None, num_shuffles)]

  if args.verbose:
    # Print out the shuffle sequence in a compact form.
    print >>sys.stderr, 'Testing shuffle sequence:'
    for s in shuffles:
      print >>sys.stderr, '  v%d%s: %s' % (width, element_type, s)
    print >>sys.stderr, ''

  # Compute a round-trip of the shuffle.
  result = range(1, width + 1)
  for s in shuffles:
    result = [result[i] if i != -1 else -1 for i in s]

  if args.verbose:
    print >>sys.stderr, 'Which transforms:'
    print >>sys.stderr, '  from: %s' % (range(1, width + 1),)
    print >>sys.stderr, '  into: %s' % (result,)
    print >>sys.stderr, ''

  # The IR uses silly names for floating point types. We also need a same-size
  # integer type.
  integral_element_type = element_type
  if element_type == 'f32':
    integral_element_type = 'i32'
    element_type = 'float'
  elif element_type == 'f64':
    integral_element_type = 'i64'
    element_type = 'double'

  # Now we need to generate IR for the shuffle function.
  subst = {'N': width, 'T': element_type, 'IT': integral_element_type}
  print """
define internal <%(N)d x %(T)s> @test(<%(N)d x %(T)s> %%v) noinline nounwind {
entry:""" % subst

  for i, s in enumerate(shuffles):
    print """
  %%s%(i)d = shufflevector <%(N)d x %(T)s> %(I)s, <%(N)d x %(T)s> undef, <%(N)d x i32> <%(S)s>
""".strip() % dict(subst,
                i=i,
                I=('%%s%d' % (i - 1)) if i != 0 else '%v',
                S=', '.join(['i32 %s' % (str(si) if si != -1 else 'undef',)
                             for si in s]))

  print """
  ret <%(N)d x %(T)s> %%s%(i)d
}
""" % dict(subst, i=len(shuffles) - 1)

  # Generate some string constants that we can use to report errors.
  for i, r in enumerate(result):
    if r != -1:
      s = ('FAIL(%(seed)s): lane %(lane)d, expected %(result)d, found %%d\\0A' %
           {'seed': args.seed, 'lane': i, 'result': r})
      s += ''.join(['\\00' for _ in itertools.repeat(None, 64 - len(s) + 2)])
      print """
@error.%(i)d = private unnamed_addr global [64 x i8] c"%(s)s"
""".strip() % {'i': i, 's': s}

  # Finally, generate a main function which will trap if any lanes are mapped
  # incorrectly (in an observable way).
  print """
define i32 @main() optnone noinline {
entry:
  ; Create a scratch space to print error messages.
  %%str = alloca [64 x i8]
  %%str.ptr = getelementptr inbounds [64 x i8]* %%str, i32 0, i32 0

  ; Build the input vector and call the test function.
  %%input = bitcast <%(N)d x %(IT)s> <%(input)s> to <%(N)d x %(T)s>
  %%v = call <%(N)d x %(T)s> @test(<%(N)d x %(T)s> %%input)
  ; We need to cast this back to an integer type vector to easily check the
  ; result.
  %%v.cast = bitcast <%(N)d x %(T)s> %%v to <%(N)d x %(IT)s>
  br label %%test.0
""" % dict(subst,
           input=', '.join(['%(IT)s %(i)s' % dict(subst, i=i)
                            for i in xrange(1, width + 1)]),
           result=', '.join(['%(IT)s %(i)s' % dict(subst,
                                                   i=i if i != -1 else 'undef')
                             for i in result]))

  # Test that each non-undef result lane contains the expected value.
  for i, r in enumerate(result):
    if r == -1:
      print """
test.%(i)d:
  ; Skip this lane, its value is undef.
  br label %%test.%(next_i)d
""" % dict(subst, i=i, next_i=i + 1)
    else:
      print """
test.%(i)d:
  %%v.%(i)d = extractelement <%(N)d x %(IT)s> %%v.cast, i32 %(i)d
  %%cmp.%(i)d = icmp ne %(IT)s %%v.%(i)d, %(r)d
  br i1 %%cmp.%(i)d, label %%die.%(i)d, label %%test.%(next_i)d

die.%(i)d:
  ; Capture the actual value and print an error message.
  %%tmp.%(i)d = zext %(IT)s %%v.%(i)d to i2048
  %%bad.%(i)d = trunc i2048 %%tmp.%(i)d to i32
  call i32 (i8*, i8*, ...)* @sprintf(i8* %%str.ptr, i8* getelementptr inbounds ([64 x i8]* @error.%(i)d, i32 0, i32 0), i32 %%bad.%(i)d)
  %%length.%(i)d = call i32 @strlen(i8* %%str.ptr)
  %%size.%(i)d = add i32 %%length.%(i)d, 1
  call i32 @write(i32 2, i8* %%str.ptr, i32 %%size.%(i)d)
  call void @llvm.trap()
  unreachable
""" % dict(subst, i=i, next_i=i + 1, r=r)

  print """
test.%d:
  ret i32 0
}

declare i32 @strlen(i8*)
declare i32 @write(i32, i8*, i32)
declare i32 @sprintf(i8*, i8*, ...)
declare void @llvm.trap() noreturn nounwind
""" % (len(result),)

if __name__ == '__main__':
  main()
