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


template<int i, int end>
struct tuple_for_each_impl
{
  template<typename Tuple, typename Fun>
  static inline __host__ __device__
  Fun do_it(Tuple& t, Fun f)
  {
    f(thrust::get<i>(t));
    return tuple_for_each_impl<i+1,end>::do_it(t, f);
  }
};


template<int i>
struct tuple_for_each_impl<i,i>
{
  template<typename Tuple, typename Fun>
  static inline __host__ __device__
  Fun do_it(Tuple&, Fun f)
  {
    return f;
  }
};


template<typename Tuple, typename Fun>
inline __host__ __device__
Fun tuple_for_each(Tuple& t, Fun f)
{ 
  return tuple_for_each_impl<0,thrust::tuple_size<Tuple>::value>::do_it(t,f);
} // end tuple_for_each()


} // end detail
} // end thrust

