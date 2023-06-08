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
    // Check the rank
    EXPECT_EQ( ( negate_subtensor.rank() ), 3 );
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

  TEST( CONJUGATE, DR_TENSOR_REAL )
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
    // Conjugate the tensor
    auto conjugate_tensor = conj( tensor, 0, 2 );
    // Access elements from const tensor
    auto real_val1 = LINALG_DETAIL::access( conjugate_tensor, 0, 0, 0 ).real();
    auto real_val2 = LINALG_DETAIL::access( conjugate_tensor, 0, 0, 1 ).real();
    auto real_val3 = LINALG_DETAIL::access( conjugate_tensor, 0, 1, 0 ).real();
    auto real_val4 = LINALG_DETAIL::access( conjugate_tensor, 0, 1, 1 ).real();
    auto real_val5 = LINALG_DETAIL::access( conjugate_tensor, 1, 0, 0 ).real();
    auto real_val6 = LINALG_DETAIL::access( conjugate_tensor, 1, 0, 1 ).real();
    auto real_val7 = LINALG_DETAIL::access( conjugate_tensor, 1, 1, 0 ).real();
    auto real_val8 = LINALG_DETAIL::access( conjugate_tensor, 1, 1, 1 ).real();
    auto imag_val1 = LINALG_DETAIL::access( conjugate_tensor, 0, 0, 0 ).imag();
    auto imag_val2 = LINALG_DETAIL::access( conjugate_tensor, 0, 0, 1 ).imag();
    auto imag_val3 = LINALG_DETAIL::access( conjugate_tensor, 0, 1, 0 ).imag();
    auto imag_val4 = LINALG_DETAIL::access( conjugate_tensor, 0, 1, 1 ).imag();
    auto imag_val5 = LINALG_DETAIL::access( conjugate_tensor, 1, 0, 0 ).imag();
    auto imag_val6 = LINALG_DETAIL::access( conjugate_tensor, 1, 0, 1 ).imag();
    auto imag_val7 = LINALG_DETAIL::access( conjugate_tensor, 1, 1, 0 ).imag();
    auto imag_val8 = LINALG_DETAIL::access( conjugate_tensor, 1, 1, 1 ).imag();
    // Check the conjugate tensor provides the correct values
    EXPECT_EQ( real_val1, 1.0 );
    EXPECT_EQ( real_val2, 5.0 );
    EXPECT_EQ( real_val3, 3.0 );
    EXPECT_EQ( real_val4, 7.0 );
    EXPECT_EQ( real_val5, 2.0 );
    EXPECT_EQ( real_val6, 6.0 );
    EXPECT_EQ( real_val7, 4.0 );
    EXPECT_EQ( real_val8, 8.0 );
    EXPECT_EQ( imag_val1, 0.0 );
    EXPECT_EQ( imag_val2, 0.0 );
    EXPECT_EQ( imag_val3, 0.0 );
    EXPECT_EQ( imag_val4, 0.0 );
    EXPECT_EQ( imag_val5, 0.0 );
    EXPECT_EQ( imag_val6, 0.0 );
    EXPECT_EQ( imag_val7, 0.0 );
    EXPECT_EQ( imag_val8, 0.0 );
    // Check the extents
    EXPECT_EQ( ( conjugate_tensor.extent( 0 ) ), 2 );
    EXPECT_EQ( ( conjugate_tensor.extent( 1 ) ), 2 );
    EXPECT_EQ( ( conjugate_tensor.extent( 2 ) ), 2 );
    EXPECT_EQ( ( conjugate_tensor.extents().extent( 0 ) ), 2 );
    EXPECT_EQ( ( conjugate_tensor.extents().extent( 1 ) ), 2 );
    EXPECT_EQ( ( conjugate_tensor.extents().extent( 2 ) ), 2 );
    // Check the rank
    EXPECT_EQ( ( conjugate_tensor.rank() ), 3 );
  }

  TEST( CONJUGATE, DR_TENSOR_COMPLEX )
  {
    using tensor_type = LINALG::dyn_tensor< ::std::complex< double >, 3 >;
    // Construct
    tensor_type tensor{ ::std::extents< ::std::size_t, 2, 2, 2 >() };
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = ::std::complex( 1.0, 1.0 );
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = ::std::complex( 2.0, 2.0 );
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = ::std::complex( 3.0, 3.0 );
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = ::std::complex( 4.0, 4.0 );
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = ::std::complex( 5.0, 5.0 );
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = ::std::complex( 6.0, 6.0 );
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = ::std::complex( 7.0, 7.0 );
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = ::std::complex( 8.0, 8.0 );
    // Conjugate the tensor
    auto conjugate_tensor = conj( tensor, 0, 2 );
    // Access elements from const tensor
    auto real_val1 = LINALG_DETAIL::access( conjugate_tensor, 0, 0, 0 ).real();
    auto real_val2 = LINALG_DETAIL::access( conjugate_tensor, 0, 0, 1 ).real();
    auto real_val3 = LINALG_DETAIL::access( conjugate_tensor, 0, 1, 0 ).real();
    auto real_val4 = LINALG_DETAIL::access( conjugate_tensor, 0, 1, 1 ).real();
    auto real_val5 = LINALG_DETAIL::access( conjugate_tensor, 1, 0, 0 ).real();
    auto real_val6 = LINALG_DETAIL::access( conjugate_tensor, 1, 0, 1 ).real();
    auto real_val7 = LINALG_DETAIL::access( conjugate_tensor, 1, 1, 0 ).real();
    auto real_val8 = LINALG_DETAIL::access( conjugate_tensor, 1, 1, 1 ).real();
    auto imag_val1 = LINALG_DETAIL::access( conjugate_tensor, 0, 0, 0 ).imag();
    auto imag_val2 = LINALG_DETAIL::access( conjugate_tensor, 0, 0, 1 ).imag();
    auto imag_val3 = LINALG_DETAIL::access( conjugate_tensor, 0, 1, 0 ).imag();
    auto imag_val4 = LINALG_DETAIL::access( conjugate_tensor, 0, 1, 1 ).imag();
    auto imag_val5 = LINALG_DETAIL::access( conjugate_tensor, 1, 0, 0 ).imag();
    auto imag_val6 = LINALG_DETAIL::access( conjugate_tensor, 1, 0, 1 ).imag();
    auto imag_val7 = LINALG_DETAIL::access( conjugate_tensor, 1, 1, 0 ).imag();
    auto imag_val8 = LINALG_DETAIL::access( conjugate_tensor, 1, 1, 1 ).imag();
    // Check the conjugate tensor provides the correct values
    EXPECT_EQ( real_val1, 1.0 );
    EXPECT_EQ( real_val2, 5.0 );
    EXPECT_EQ( real_val3, 3.0 );
    EXPECT_EQ( real_val4, 7.0 );
    EXPECT_EQ( real_val5, 2.0 );
    EXPECT_EQ( real_val6, 6.0 );
    EXPECT_EQ( real_val7, 4.0 );
    EXPECT_EQ( real_val8, 8.0 );
    EXPECT_EQ( imag_val1, -1.0 );
    EXPECT_EQ( imag_val2, -5.0 );
    EXPECT_EQ( imag_val3, -3.0 );
    EXPECT_EQ( imag_val4, -7.0 );
    EXPECT_EQ( imag_val5, -2.0 );
    EXPECT_EQ( imag_val6, -6.0 );
    EXPECT_EQ( imag_val7, -4.0 );
    EXPECT_EQ( imag_val8, -8.0 );
    // Check the extents
    EXPECT_EQ( ( conjugate_tensor.extent( 0 ) ), 2 );
    EXPECT_EQ( ( conjugate_tensor.extent( 1 ) ), 2 );
    EXPECT_EQ( ( conjugate_tensor.extent( 2 ) ), 2 );
    EXPECT_EQ( ( conjugate_tensor.extents().extent( 0 ) ), 2 );
    EXPECT_EQ( ( conjugate_tensor.extents().extent( 1 ) ), 2 );
    EXPECT_EQ( ( conjugate_tensor.extents().extent( 2 ) ), 2 );
    // Check the rank
    EXPECT_EQ( ( conjugate_tensor.rank() ), 3 );
  }

  TEST( CONJUGATE, FS_TENSOR_COMPLEX )
  {
    
    using tensor_type = LINALG::fs_tensor< ::std::complex< double >, ::std::extents< ::std::size_t, 2, 2, 2 > >;
    // Construct
    tensor_type tensor { };
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0, 0, 0 ) = ::std::complex( 1.0, 1.0 );
    LINALG_DETAIL::access( tensor, 0, 0, 1 ) = ::std::complex( 2.0, 2.0 );
    LINALG_DETAIL::access( tensor, 0, 1, 0 ) = ::std::complex( 3.0, 3.0 );
    LINALG_DETAIL::access( tensor, 0, 1, 1 ) = ::std::complex( 4.0, 4.0 );
    LINALG_DETAIL::access( tensor, 1, 0, 0 ) = ::std::complex( 5.0, 5.0 );
    LINALG_DETAIL::access( tensor, 1, 0, 1 ) = ::std::complex( 6.0, 6.0 );
    LINALG_DETAIL::access( tensor, 1, 1, 0 ) = ::std::complex( 7.0, 7.0 );
    LINALG_DETAIL::access( tensor, 1, 1, 1 ) = ::std::complex( 8.0, 8.0 );
    // Conjugate the tensor
    auto conjugate_tensor = conj( tensor, 0, 2 );
    // Access elements from const tensor
    auto real_val1 = LINALG_DETAIL::access( conjugate_tensor, 0, 0, 0 ).real();
    auto real_val2 = LINALG_DETAIL::access( conjugate_tensor, 0, 0, 1 ).real();
    auto real_val3 = LINALG_DETAIL::access( conjugate_tensor, 0, 1, 0 ).real();
    auto real_val4 = LINALG_DETAIL::access( conjugate_tensor, 0, 1, 1 ).real();
    auto real_val5 = LINALG_DETAIL::access( conjugate_tensor, 1, 0, 0 ).real();
    auto real_val6 = LINALG_DETAIL::access( conjugate_tensor, 1, 0, 1 ).real();
    auto real_val7 = LINALG_DETAIL::access( conjugate_tensor, 1, 1, 0 ).real();
    auto real_val8 = LINALG_DETAIL::access( conjugate_tensor, 1, 1, 1 ).real();
    auto imag_val1 = LINALG_DETAIL::access( conjugate_tensor, 0, 0, 0 ).imag();
    auto imag_val2 = LINALG_DETAIL::access( conjugate_tensor, 0, 0, 1 ).imag();
    auto imag_val3 = LINALG_DETAIL::access( conjugate_tensor, 0, 1, 0 ).imag();
    auto imag_val4 = LINALG_DETAIL::access( conjugate_tensor, 0, 1, 1 ).imag();
    auto imag_val5 = LINALG_DETAIL::access( conjugate_tensor, 1, 0, 0 ).imag();
    auto imag_val6 = LINALG_DETAIL::access( conjugate_tensor, 1, 0, 1 ).imag();
    auto imag_val7 = LINALG_DETAIL::access( conjugate_tensor, 1, 1, 0 ).imag();
    auto imag_val8 = LINALG_DETAIL::access( conjugate_tensor, 1, 1, 1 ).imag();
    // Check the conjugate tensor provides the correct values
    EXPECT_EQ( real_val1, 1.0 );
    EXPECT_EQ( real_val2, 5.0 );
    EXPECT_EQ( real_val3, 3.0 );
    EXPECT_EQ( real_val4, 7.0 );
    EXPECT_EQ( real_val5, 2.0 );
    EXPECT_EQ( real_val6, 6.0 );
    EXPECT_EQ( real_val7, 4.0 );
    EXPECT_EQ( real_val8, 8.0 );
    EXPECT_EQ( imag_val1, -1.0 );
    EXPECT_EQ( imag_val2, -5.0 );
    EXPECT_EQ( imag_val3, -3.0 );
    EXPECT_EQ( imag_val4, -7.0 );
    EXPECT_EQ( imag_val5, -2.0 );
    EXPECT_EQ( imag_val6, -6.0 );
    EXPECT_EQ( imag_val7, -4.0 );
    EXPECT_EQ( imag_val8, -8.0 );
    // Check the extents
    EXPECT_EQ( ( conjugate_tensor.extent( 0 ) ), 2 );
    EXPECT_EQ( ( conjugate_tensor.extent( 1 ) ), 2 );
    EXPECT_EQ( ( conjugate_tensor.extent( 2 ) ), 2 );
    EXPECT_EQ( ( conjugate_tensor.extents().extent( 0 ) ), 2 );
    EXPECT_EQ( ( conjugate_tensor.extents().extent( 1 ) ), 2 );
    EXPECT_EQ( ( conjugate_tensor.extents().extent( 2 ) ), 2 );
    // Check the rank
    EXPECT_EQ( ( conjugate_tensor.rank() ), 3 );
  }

  TEST( CONJUGATE, TENSOR_VIEW_REAL )
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
    // Conjugate the tensor
    auto conjugate_tensor = conj( subtensor );
    // Access elements from conj tensor
    auto real_val1 = LINALG_DETAIL::access( conjugate_tensor, 0, 0 ).real();
    auto imag_val1 = LINALG_DETAIL::access( conjugate_tensor, 0, 0 ).imag();
    auto real_val2 = LINALG_DETAIL::access( conjugate_tensor, 0, 1 ).real();
    auto imag_val2 = LINALG_DETAIL::access( conjugate_tensor, 0, 1 ).imag();
    // Check the tensor copy was populated correctly and provided the correct values
    EXPECT_EQ( real_val1, 1.0 );
    EXPECT_EQ( imag_val1, 0.0 );
    EXPECT_EQ( real_val2, 5.0 );
    EXPECT_EQ( imag_val2, 0.0 );
    // Check the extents
    EXPECT_EQ( ( conjugate_tensor.extent( 0 ) ), 1 );
    EXPECT_EQ( ( conjugate_tensor.extent( 1 ) ), 2 );
    EXPECT_EQ( ( conjugate_tensor.extents().extent( 0 ) ), 1 );
    EXPECT_EQ( ( conjugate_tensor.extents().extent( 1 ) ), 2 );
  }

  TEST( CONJUGATE, DR_VECTOR_REAL )
  {
    using tensor_type = LINALG::dyn_vector< double >;
    // Construct
    tensor_type tensor{ 2 };
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0 ) = 1.0;
    LINALG_DETAIL::access( tensor, 1 ) = 2.0;
    // Conjugate the tensor
    auto conjugate_tensor = conj( tensor );
    // Access elements from conjugate tensor
    auto real_val1 = LINALG_DETAIL::access( conjugate_tensor, 0 ).real();
    auto imag_val1 = LINALG_DETAIL::access( conjugate_tensor, 0 ).imag();
    auto real_val2 = LINALG_DETAIL::access( conjugate_tensor, 1 ).real();
    auto imag_val2 = LINALG_DETAIL::access( conjugate_tensor, 1 ).imag();
    // Check the tensor copy was populated correctly and provided the correct values
    EXPECT_EQ( real_val1, 1.0 );
    EXPECT_EQ( imag_val1, 0.0 );
    EXPECT_EQ( real_val2, 2.0 );
    EXPECT_EQ( imag_val2, 0.0 );
    // Check the extents
    EXPECT_EQ( ( conjugate_tensor.extent( 0 ) ), 2 );
    EXPECT_EQ( ( conjugate_tensor.extents().extent( 0 ) ), 2 );
  }

  TEST( CONJUGATE, DR_VECTOR_COMPLEX )
  {
    using tensor_type = LINALG::dyn_vector< ::std::complex< double > >;
    // Construct
    tensor_type tensor{ 2 };
    // Populate via mutable index access
    LINALG_DETAIL::access( tensor, 0 ) = ::std::complex( 1.0, 1.0 );
    LINALG_DETAIL::access( tensor, 1 ) = ::std::complex( 2.0, 2.0 );
    // Conjugate the tensor
    auto conjugate_tensor = conj( tensor );
    // Access elements from conjugate tensor
    auto real_val1 = LINALG_DETAIL::access( conjugate_tensor, 0 ).real();
    auto imag_val1 = LINALG_DETAIL::access( conjugate_tensor, 0 ).imag();
    auto real_val2 = LINALG_DETAIL::access( conjugate_tensor, 1 ).real();
    auto imag_val2 = LINALG_DETAIL::access( conjugate_tensor, 1 ).imag();
    // Check the tensor copy was populated correctly and provided the correct values
    EXPECT_EQ( real_val1, 1.0 );
    EXPECT_EQ( imag_val1, -1.0 );
    EXPECT_EQ( real_val2, 2.0 );
    EXPECT_EQ( imag_val2, -2.0 );
    // Check the extents
    EXPECT_EQ( ( conjugate_tensor.extent( 0 ) ), 2 );
    EXPECT_EQ( ( conjugate_tensor.extents().extent( 0 ) ), 2 );
  }
}