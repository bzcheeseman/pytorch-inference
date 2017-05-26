#! /bin/bash
#  assert.sh

cd ../cmake-build-debug
for test in test_*; do
    if [ "$test" == "test_*.dat" ] ; then
        continue;
    fi
    echo "Testing: $test"
    ./"$test" > /dev/null
    echo "Testing status: $?"
done
rm test_*.dat