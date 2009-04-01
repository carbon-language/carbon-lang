// RUN: clang-cc -emit-llvm -o %t %s

@class XCLogItemAdapter;
@class XCEConfigurableDataSource;

@interface XCBuildResultsOutlineLogic  {
    XCLogItemAdapter * _toplevelItemAdapter;
}

@end


@interface XCBuildResultsOutlineView  
@property(nonatomic,readonly) XCEConfigurableDataSource * xceDataSource;

@end

@interface XCEConfigurableDataSource 
@property (nonatomic, assign) id context;

@end


@implementation XCBuildResultsOutlineView
@dynamic xceDataSource;
- selectionSource {
    XCBuildResultsOutlineLogic * outlineLogic; 
}

@end

@interface XCLogItemAdapter { 
    id _textColor;
}


@end

@implementation XCLogItemAdapter
- (id) FOOO
{
    return _textColor;
}
