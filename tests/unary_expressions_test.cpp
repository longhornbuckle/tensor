#include <gtest/gtest.h>
#include <experimental/linear_algebra.hpp>

namespace
{

  TEST( NEGATION, DR_TENSOR )
  {
    using tensor_type = LINALG::dyn_tensor< double, 3 >;
    // Construct
    tensor_type tensor{ ::std::extents< ::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8.0;
    // Negate the tensor
    auto negate_tensor { -tensor };
    // Access elements from const tensor
    auto val1 = LINALG_DETAIL::access( negate_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( negate_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( negate_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( negate_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( negate_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( negate_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( negate_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( negate_tensor, 1, 1, 1 );
    // Check the tensor copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, -1.0 );
    EXPECT_EQ( val2, -2.0 );
    EXPECT_EQ( val3, -3.0 );
    EXPECT_EQ( val4, -4.0 );
    EXPECT_EQ( val5, -5.0 );
    EXPECT_EQ( val6, -6.0 );
    EXPECT_EQ( val7, -7.0 );
    EXPECT_EQ( val8, -8.0 );
  }

  TEST( NEGATION, FS_TENSOR )
  {
    using tensor_type = LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 3, 3, 3 >, ::std::layout_right, ::std::default_accessor<double> >;
    // Construct
    tensor_type tensor { };
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8.0;
    // Negate the tensor
    auto negate_tensor { -tensor };
    // Access elements from const tensor
    auto val1 = LINALG_DETAIL::access( negate_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( negate_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( negate_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( negate_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( negate_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( negate_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( negate_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( negate_tensor, 1, 1, 1 );
    // Check the tensor copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, -1.0 );
    EXPECT_EQ( val2, -2.0 );
    EXPECT_EQ( val3, -3.0 );
    EXPECT_EQ( val4, -4.0 );
    EXPECT_EQ( val5, -5.0 );
    EXPECT_EQ( val6, -6.0 );
    EXPECT_EQ( val7, -7.0 );
    EXPECT_EQ( val8, -8.0 );
  }

  TEST( NEGATION, TENSOR_VIEW )
  {
    // Get a rank 3 subtensor
    using fs_tensor_type = LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 5, 5, 5 >, ::std::layout_right, ::std::default_accessor<double> >;
    // Default construct
    fs_tensor_type fs_tensor;
    double val = 1;
    for ( auto i : { 0, 1, 2, 3, 4 } )
    {
      for ( auto j : { 0, 1, 2, 3, 4 } )
      {
        for ( auto k : { 0, 1, 2, 3, 4 } )
        {
          LINALG_DETAIL::access( fs_tensor, i, j, k ) = val;
          val = 2 * val;
        }
      }
    }
    const fs_tensor_type& const_fs_tensor( fs_tensor );
    auto subtensor = LINALG::subtensor( const_fs_tensor, ::std::tuple(2,5), ::std::tuple(2,4), ::std::tuple(2,4) );
    // Negate subtensor
    auto negate_subtensor = - subtensor;
    // Check negated values
    EXPECT_EQ( ( LINALG_DETAIL::access( negate_subtensor, 0, 0, 0 ) ), ( -LINALG_DETAIL::access( subtensor, 0, 0, 0 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( negate_subtensor, 1, 0, 0 ) ), ( -LINALG_DETAIL::access( subtensor, 1, 0, 0 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( negate_subtensor, 2, 0, 0 ) ), ( -LINALG_DETAIL::access( subtensor, 2, 0, 0 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( negate_subtensor, 0, 1, 0 ) ), ( -LINALG_DETAIL::access( subtensor, 0, 1, 0 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( negate_subtensor, 1, 1, 0 ) ), ( -LINALG_DETAIL::access( subtensor, 1, 1, 0 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( negate_subtensor, 2, 1, 0 ) ), ( -LINALG_DETAIL::access( subtensor, 2, 1, 0 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( negate_subtensor, 0, 0, 1 ) ), ( -LINALG_DETAIL::access( subtensor, 0, 0, 1 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( negate_subtensor, 1, 0, 1 ) ), ( -LINALG_DETAIL::access( subtensor, 1, 0, 1 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( negate_subtensor, 2, 0, 1 ) ), ( -LINALG_DETAIL::access( subtensor, 2, 0, 1 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( negate_subtensor, 0, 1, 1 ) ), ( -LINALG_DETAIL::access( subtensor, 0, 1, 1 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( negate_subtensor, 1, 1, 1 ) ), ( -LINALG_DETAIL::access( subtensor, 1, 1, 1 ) ) );
    EXPECT_EQ( ( LINALG_DETAIL::access( negate_subtensor, 2, 1, 1 ) ), ( -LINALG_DETAIL::access( subtensor, 2, 1, 1 ) ) );
    // Check the extents
    EXPECT_EQ( ( negate_subtensor.extent( 0 ) ), 3 );
    EXPECT_EQ( ( negate_subtensor.extent( 1 ) ), 2 );
    EXPECT_EQ( ( negate_subtensor.extent( 2 ) ), 2 );
    EXPECT_EQ( ( negate_subtensor.extents().extent( 0 ) ), 3 );
    EXPECT_EQ( ( negate_subtensor.extents().extent( 1 ) ), 2 );
    EXPECT_EQ( ( negate_subtensor.extents().extent( 2 ) ), 2 );
  }

  TEST( TRANSPOSE, DR_TENSOR )
  {
    using tensor_type = LINALG::dyn_tensor< double, 3 >;
    // Construct
    tensor_type tensor{ ::std::extents< ::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8.0;
    // Transpose the tensor
    auto transpose_tensor = trans( tensor, 0, 2 );
    // Access elements from const tensor
    auto val1 = LINALG_DETAIL::access( transpose_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( transpose_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( transpose_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( transpose_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( transpose_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( transpose_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( transpose_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( transpose_tensor, 1, 1, 1 );
    // Check the tensor copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 5.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 7.0 );
    EXPECT_EQ( val5, 2.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 4.0 );
    EXPECT_EQ( val8, 8.0 );
  }

  TEST( TRANSPOSE, FS_TENSOR )
  {
    using tensor_type = LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 > >;
    // Construct
    tensor_type tensor;
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8.0;
    // Transpose the tensor
    auto transpose_tensor = trans( tensor, 0, 2 );
    // Access elements from const tensor
    auto val1 = LINALG_DETAIL::access( transpose_tensor, 0, 0, 0 );
    auto val2 = LINALG_DETAIL::access( transpose_tensor, 0, 0, 1 );
    auto val3 = LINALG_DETAIL::access( transpose_tensor, 0, 1, 0 );
    auto val4 = LINALG_DETAIL::access( transpose_tensor, 0, 1, 1 );
    auto val5 = LINALG_DETAIL::access( transpose_tensor, 1, 0, 0 );
    auto val6 = LINALG_DETAIL::access( transpose_tensor, 1, 0, 1 );
    auto val7 = LINALG_DETAIL::access( transpose_tensor, 1, 1, 0 );
    auto val8 = LINALG_DETAIL::access( transpose_tensor, 1, 1, 1 );
    // Check the tensor copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 5.0 );
    EXPECT_EQ( val3, 3.0 );
    EXPECT_EQ( val4, 7.0 );
    EXPECT_EQ( val5, 2.0 );
    EXPECT_EQ( val6, 6.0 );
    EXPECT_EQ( val7, 4.0 );
    EXPECT_EQ( val8, 8.0 );
    // Check the extents
    EXPECT_EQ( ( transpose_tensor.extent( 0 ) ), 2 );
    EXPECT_EQ( ( transpose_tensor.extent( 1 ) ), 2 );
    EXPECT_EQ( ( transpose_tensor.extent( 2 ) ), 2 );
    EXPECT_EQ( ( transpose_tensor.extents().extent( 0 ) ), 2 );
    EXPECT_EQ( ( transpose_tensor.extents().extent( 1 ) ), 2 );
    EXPECT_EQ( ( transpose_tensor.extents().extent( 2 ) ), 2 );
  }

  TEST( TRANSPOSE, TENSOR_VIEW )
  {
    using tensor_type = LINALG::fs_tensor< double, ::std::extents< ::std::size_t, 2, 2, 2 > >;
    // Construct
    tensor_type tensor;
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = 2.0;
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = 3.0;
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = 4.0;
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = 5.0;
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = 6.0;
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = 7.0;
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = 8.0;
    // Get a view
    auto subtensor = LINALG::subtensor( tensor, ::std::tuple( 0, 2 ), ::std::tuple( 0, 1 ), 0 );
    // Transpose the tensor
    auto transpose_tensor = trans( subtensor );
    // Access elements from const tensor
    auto val1 = LINALG_DETAIL::access( transpose_tensor, 0, 0 );
    auto val2 = LINALG_DETAIL::access( transpose_tensor, 0, 1 );
    // Check the tensor copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 5.0 );
    // Check the extents
    EXPECT_EQ( ( transpose_tensor.extent( 0 ) ), 1 );
    EXPECT_EQ( ( transpose_tensor.extent( 1 ) ), 2 );
    EXPECT_EQ( ( transpose_tensor.extents().extent( 0 ) ), 1 );
    EXPECT_EQ( ( transpose_tensor.extents().extent( 1 ) ), 2 );
  }

  TEST( TRANSPOSE, DR_VECTOR )
  {
    using tensor_type = LINALG::dyn_vector< double >;
    // Construct
    tensor_type tensor{ 2 };
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor, 1 ) = 2.0;
    // Transpose the tensor
    auto transpose_tensor = trans( tensor );
    // Access elements from const tensor
    auto val1 = LINALG_DETAIL::access( transpose_tensor, 0 );
    auto val2 = LINALG_DETAIL::access( transpose_tensor, 1 );
    // Check the tensor copy was populated correctly and provided the correct values
    EXPECT_EQ( val1, 1.0 );
    EXPECT_EQ( val2, 2.0 );
    // Check the extents
    EXPECT_EQ( ( transpose_tensor.extent( 0 ) ), 2 );
    EXPECT_EQ( ( transpose_tensor.extents().extent( 0 ) ), 2 );
  }

}