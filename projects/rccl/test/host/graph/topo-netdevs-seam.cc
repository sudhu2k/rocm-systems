/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// The real src/graph/topo.cc for binaries that set its NCCL_NETDEVS_POLICY statics without compiling topo-test.cc,
// which defines the same setter.

#include "fakes/graph_fakes.h"
#include "fakes/param_redirect.h"

#include TOPO_CC_PATH

void TopoSetNetDevsPolicy(enum netDevsPolicy policy, int num) {
  int unused;
  ncclTopoGetNetDevsPolicy(nullptr, &unused);  // consume the call_once first
  netDevsPolicy = policy;
  netDevsPolicyNum = num;
}
