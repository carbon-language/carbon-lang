// RUN: %clang_cc1  -fsyntax-only -fobjc-default-synthesize-properties -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -fobjc-default-synthesize-properties -verify -Wno-objc-root-class %s
// rdar://13178483

@class NSMutableDictionary;

@interface LaunchdJobs 

@property (nonatomic,retain) NSMutableDictionary *uuids_jobs; // expected-note 2 {{'_uuids_jobs' declared here}}

@end

@implementation LaunchdJobs

-(void)job
{

 [uuids_jobs objectForKey]; // expected-error {{use of undeclared identifier 'uuids_jobs'}} \
                            // expected-warning {{instance method '-objectForKey' not found}}
}


@end

void
doLaunchdJobCPU()
{
 [uuids_jobs enumerateKeysAndObjectsUsingBlock]; // expected-error {{use of undeclared identifier 'uuids_jobs'}}
}
