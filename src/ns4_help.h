#ifndef NS4_HELP_H
#define NS4_HELP_H

#include <stdio.h>

typedef enum {
    NS4_HELP_NS4,
    NS4_HELP_SERVER,
    NS4_HELP_AGENT,
    NS4_HELP_BENCH,
    NS4_HELP_EVAL,
} ns4_help_tool;

void ns4_help_print(FILE *fp, ns4_help_tool tool, const char *topic);

#endif
