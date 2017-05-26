#! /bin/bash
#  assert.sh

QUIET=false

cd ../cmake-build-debug
for test in test_*; do

    if [ "$test" == "*.dat" ] ; then
        continue;
    fi

    echo "Testing: $test"

    if [ "$QUIET" == true ] ; then
        ./"$test" > /dev/null
    else
        ./"$test"
    fi

    if [ "$?" == "0" ] ; then
        echo "Testing status: Passed - ✓"
        rm *.dat
    else
        echo "Testing status: Failed - ✗"
    fi

done