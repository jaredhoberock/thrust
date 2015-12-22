/*
 *  Copyright 2008-2013 NVIDIA Corporation
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
#include <thrust/detail/type_traits.h>

namespace thrust
{
namespace detail
{


// define apply2 for tuple_meta_accumulate_impl
template<typename UnaryMetaFunctionClass, class Arg1, class Arg2>
  struct apply2
    : UnaryMetaFunctionClass::template apply<Arg1,Arg2>
{
}; // end apply2


// Meta-accumulate algorithm for tuples. Note: The template 
// parameter StartType corresponds to the initial value in 
// ordinary accumulation.
//
template<class Tuple, class BinaryMetaFun, class StartType>
  struct tuple_meta_accumulate;

template<
    typename Tuple
  , class BinaryMetaFun
  , typename StartType
>
  struct tuple_meta_accumulate_impl
{
   typedef typename apply2<
       BinaryMetaFun
     , typename Tuple::head_type
     , typename tuple_meta_accumulate<
           typename Tuple::tail_type
         , BinaryMetaFun
         , StartType 
       >::type
   >::type type;
};


template<
    typename Tuple
  , class BinaryMetaFun
  , typename StartType
>
struct tuple_meta_accumulate
  : thrust::detail::eval_if<
        thrust::detail::is_same<Tuple, thrust::null_type>::value
      , thrust::detail::identity_<StartType>
      , tuple_meta_accumulate_impl<
            Tuple
          , BinaryMetaFun
          , StartType
        >
    > // end eval_if
{
}; // end tuple_meta_accumulate


} // end detail
} // end thrust

