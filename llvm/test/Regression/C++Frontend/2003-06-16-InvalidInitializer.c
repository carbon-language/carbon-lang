typedef struct {
    char *auth_pwfile;
} auth_config_rec;

void *Ptr = &((auth_config_rec*)0)->auth_pwfile;

int main() { return 0; }
