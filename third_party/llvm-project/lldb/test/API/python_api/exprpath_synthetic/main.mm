#import <Foundation/Foundation.h>
#include <vector>

int main (int argc, char const *argv[])
{
    std::vector<int> v{1,2,3,4,5};
    NSArray *a = @[@"Hello",@"World",@"From Me"];
    return 0; //% v = self.frame().FindVariable("v"); v0 = v.GetChildAtIndex(0); s = lldb.SBStream(); v0.GetExpressionPath(s);
    //% self.runCmd("expr %s = 12" % s.GetData()); self.assertTrue(v0.GetValueAsUnsigned() == 12, "value change via expr failed")
    //% a = self.frame().FindVariable("a"); a1 = a.GetChildAtIndex(1); s = lldb.SBStream(); a1.GetExpressionPath(s);
    //% self.expect("po %s" % s.GetData(), substrs = ["World"])
}
