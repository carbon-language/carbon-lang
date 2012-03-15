#include <AvailabilityMacros.h>

#if !defined(MAC_OS_X_VERSION_10_7) || MAC_OS_X_VERSION_MAX_ALLOWED < MAC_OS_X_VERSION_10_7
#define BUILDING_ON_SNOW_LEOPARD 1
#endif

#if !BUILDING_ON_SNOW_LEOPARD
#define __XPC_PRIVATE_H__
#include <xpc/xpc.h>
#include <Security/Security.h>
#include "LauncherXPCService.h"

// Returns 0 if successful.
int _validate_authorization(xpc_object_t message)
{
	size_t data_length = 0ul;
	const char *data_bytes = (const char *)xpc_dictionary_get_data(message, LauncherXPCServiceAuthKey, &data_length);
    
	AuthorizationExternalForm extAuth;
    if (data_length < sizeof(extAuth.bytes))
        return 1;
    
	memcpy(extAuth.bytes, data_bytes, sizeof(extAuth.bytes));
    AuthorizationRef authRef;
	if (AuthorizationCreateFromExternalForm(&extAuth, &authRef) != errAuthorizationSuccess)
        return 2;
    
    AuthorizationItem item1 = { LaunchUsingXPCRightName, 0, NULL, 0 };
    AuthorizationItem items[] = {item1};
    AuthorizationRights requestedRights = {1, items };
    AuthorizationRights *outAuthorizedRights = NULL;
	OSStatus status = AuthorizationCopyRights(authRef, &requestedRights, kAuthorizationEmptyEnvironment, kAuthorizationFlagDefaults, &outAuthorizedRights);
	
	// Given a set of rights, return the subset that is currently authorized by the AuthorizationRef given; count(subset) > 0  -> success.
	bool auth_success = (status == errAuthorizationSuccess && outAuthorizedRights && outAuthorizedRights->count > 0) ? true : false;
	
	if (outAuthorizedRights) AuthorizationFreeItemSet(outAuthorizedRights);
        
        return auth_success ? 0 : 3;
}

#endif
