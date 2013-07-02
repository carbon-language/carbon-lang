# Test cases where we spill from one frame index to another, both of which
# would be out of range of MVC.  At present we don't use MVC in this case.
# RUN: python %s | llc -mtriple=s390x-linux-gnu | FileCheck %s

# There are 8 usable call-saved GPRs.  The first 160 bytes of the frame
# are needed for the ABI call frame, and a further 8 bytes are needed
# for the emergency spill slot.  That means we will have at least one
# out-of-range slot if:
#
#    count == (4096 - 168) / 8 + 8 + 1 == 500
#
# Add in some extra just to be sure.
#
# CHECK: f1:
# CHECK-NOT: mvc
# CHECK: br %r14
count = 510

print 'declare void @foo(i64 *%base0, i64 *%base1)'
print ''
print 'define void @f1() {'

for i in range(2):
    print '  %%alloc%d = alloca [%d x i64]' % (i, count / 2)
    print ('  %%base%d = getelementptr [%d x i64] * %%alloc%d, i64 0, i64 0'
           % (i, count / 2, i))

print '  call void @foo(i64 *%base0, i64 *%base1)'
print ''

for i in range(count):
    print '  %%ptr%d = getelementptr i64 *%%base%d, i64 %d' % (i, i % 2, i / 2)
    print '  %%val%d = load i64 *%%ptr%d' % (i, i)
    print ''

print '  call void @foo(i64 *%base0, i64 *%base1)'
print ''

for i in range (count):
    print '  store i64 %%val%d, i64 *%%ptr%d' % (i, i)

print ''
print '  call void @foo(i64 *%base0, i64 *%base1)'
print ''
print '  ret void'
print '}'
