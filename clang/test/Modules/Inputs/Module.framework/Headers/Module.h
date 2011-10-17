#ifndef MODULE_H
#define MODULE_H
const char *getModuleVersion(void);

#ifdef FOO
#  error Module should have been built without -DFOO
#endif

@interface Module
+(const char *)version; // retrieve module version
+alloc;
@end

#define MODULE_H_MACRO 1
#__private_macro__ MODULE_H_MACRO

#endif // MODULE_H
