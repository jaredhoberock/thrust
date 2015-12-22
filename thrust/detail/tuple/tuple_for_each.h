/*
 *  Copyright 2008-2015 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/tuple.h>

namespace thrust
{
namespace detail
{


template<typename Fun>
inline __host__ __device__
Fun tuple_for_each(thrust::null_type, Fun f)
{
  return f;
} // end tuple_for_each()


template<typename Tuple, typename Fun>
inline __host__ __device__
Fun tuple_for_each(Tuple& t, Fun f)
{ 
  f( t.get_head() );
  return thrust::detail::tuple_for_each(t.get_tail(), f);
} // end tuple_for_each()


} // end detail
} // end thrust

