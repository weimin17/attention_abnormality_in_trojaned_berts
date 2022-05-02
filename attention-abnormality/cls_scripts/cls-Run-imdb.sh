#!/bin/bash

# # compatable for imdb-mulparams-mlp-900, imdb-mulparams-mlp-150, yelp-mlp-200, sst2-mlp-200

# bash cls.sh [datasets_name] [model_root] [code_file_path] [examples_dirpath]
# # # imdb-900
# bash cls-Run-imdb.sh imdb-mulparams-mlp-900 /scr/author/author_code/GenerationData/model_zoo /scr/author/author_code/src/attn_attri_sa_v4 /scr/author/author_code/GenerationData/dev-custom-imdb
# # # yelp-200
# bash cls-Run-imdb.sh yelp-mlp-200 /scr/author/author_code/GenerationData/model_zoo /scr/author/author_code/src/attn_attri_sa_v4 /scr/author/author_code/GenerationData/dev-custom-imdb



# # 150

# # 


# # 200
declare -a arr=('id-00000000' 'id-00000001' 'id-00000002' 'id-00000003' 'id-00000004' 'id-00000005' 'id-00000006' 'id-00000007' 'id-00000008' 'id-00000009' 'id-00000010' 'id-00000011' 'id-00000012' 'id-00000013' 'id-00000014' 'id-00000015' 'id-00000016' 'id-00000017' 'id-00000018' 'id-00000019' 'id-00000020' 'id-00000021' 'id-00000022' 'id-00000023' 'id-00000024' 'id-00000025' 'id-00000026' 'id-00000027' 'id-00000028' 'id-00000029' 'id-00000030' 'id-00000031' 'id-00000032' 'id-00000033' 'id-00000034' 'id-00000035' 'id-00000036' 'id-00000037' 'id-00000038' 'id-00000039' 'id-00000040' 'id-00000041' 'id-00000042' 'id-00000043' 'id-00000044' 'id-00000045' 'id-00000046' 'id-00000047' 'id-00000048' 'id-00000049' 'id-00000050' 'id-00000051' 'id-00000052' 'id-00000053' 'id-00000054' 'id-00000055' 'id-00000056' 'id-00000057' 'id-00000058' 'id-00000059' 'id-00000060' 'id-00000061' 'id-00000062' 'id-00000063' 'id-00000064' 'id-00000065' 'id-00000066' 'id-00000067' 'id-00000068' 'id-00000069' 'id-00000070' 'id-00000071' 'id-00000072' 'id-00000073' 'id-00000074' 'id-00000075' 'id-00000076' 'id-00000077' 'id-00000078' 'id-00000079' 'id-00000080' 'id-00000081' 'id-00000082' 'id-00000083' 'id-00000084' 'id-00000085' 'id-00000086' 'id-00000087' 'id-00000088' 'id-00000089' 'id-00000090' 'id-00000091' 'id-00000092' 'id-00000093' 'id-00000094' 'id-00000095' 'id-00000096' 'id-00000097' 'id-00000098' 'id-00000099' 'id-00000100' 'id-00000101' 'id-00000102' 'id-00000103' 'id-00000104' 'id-00000105' 'id-00000106' 'id-00000107' 'id-00000108' 'id-00000109' 'id-00000110' 'id-00000111' 'id-00000112' 'id-00000113' 'id-00000114' 'id-00000115' 'id-00000116' 'id-00000117' 'id-00000118' 'id-00000119' 'id-00000120' 'id-00000121' 'id-00000122' 'id-00000123' 'id-00000124' 'id-00000125' 'id-00000126' 'id-00000127' 'id-00000128' 'id-00000129' 'id-00000130' 'id-00000131' 'id-00000132' 'id-00000133' 'id-00000134' 'id-00000135' 'id-00000136' 'id-00000137' 'id-00000138' 'id-00000139' 'id-00000140' 'id-00000141' 'id-00000142' 'id-00000143' 'id-00000144' 'id-00000145' 'id-00000146' 'id-00000147' 'id-00000148' 'id-00000149' 'id-00000150' 'id-00000151' 'id-00000152' 'id-00000153' 'id-00000154' 'id-00000155' 'id-00000156' 'id-00000157' 'id-00000158' 'id-00000159' 'id-00000160' 'id-00000161' 'id-00000162' 'id-00000163' 'id-00000164' 'id-00000165' 'id-00000166' 'id-00000167' 'id-00000168' 'id-00000169' 'id-00000170' 'id-00000171' 'id-00000172' 'id-00000173' 'id-00000174' 'id-00000175' 'id-00000176' 'id-00000177' 'id-00000178' 'id-00000179' 'id-00000180' 'id-00000181' 'id-00000182' 'id-00000183' 'id-00000184' 'id-00000185' 'id-00000186' 'id-00000187' 'id-00000188' 'id-00000189' 'id-00000190' 'id-00000191' 'id-00000192' 'id-00000193' 'id-00000194' 'id-00000195' 'id-00000196' 'id-00000197' 'id-00000198' 'id-00000199')

N=7
# COUNTER=0
for id in ${arr[@]}; do
    (
        echo "START $id .. "
        # echo $(( (RANDOM % 7) ))
        python cls_imdb.v2.py --datasets_name $1 --model_id $id --model_root $2 --root $3 --examples_dirpath $4
    ) &

    # allow to execute up to $N jobs in parallel
    if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
        # now there are $N jobs already running, so wait here for any job
        # to be finished so there is a place to start next one.
        wait -n
    fi

done

# no more jobs to be started but wait for pending jobs
# (all need to be finished)
wait

echo "all done"