. ./cmd.sh
. ./path.sh

# 101 - us 10000
# 102 - england 10000
# 103 - australia 4271
# 104 - canada 3880
# 105 - scotland 1546
# 106 - ireland 938
# 107 - wales 261

bnf_dim=1024
affix=

nj=20
train_stage=-10
srand=0
get_egs_stage=-10
decode_stage=-10
num_jobs_initial=4
num_jobs_final=7
modelDirectories=/home/abhinav/kaldi/accents/exp

. utils/parse_options.sh


dir=exp/nnet3_separate_accents/multitask_$affix


accent_list=""


[ ! -f local.conf ] && echo 'the file local.conf does not exist! Read README.txt for more details.' && exit 1;
. local.conf || exit 1;

num_langs=${#lang_list[@]}


mfccForBaseData=0
mfccForSp=0
align=0
mfccHiresForBaseData=0
mfccHiresForSp=0
ivector=0
config=0
egs=0
combineEgs=0
train=0
priors=0
decode=0
wer=1



if [ $mfccForBaseData -eq 1 ]; then
  for i in $(seq 7); do
    mfccdir=exp/mfcc/10$i
  	data="data/10"$i
  	for x in train dev; do
  		steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_16k.conf --cmd "$train_cmd" \
  				$data/$x exp/make_mfcc/$x $mfccdir
  		steps/compute_cmvn_stats.sh $data/$x exp/make_mfcc/$x $mfccdir
  		utils/fix_data_dir.sh $data/$x
  	done
  done
fi

if [ $mfccForSp -eq 1 ]; then
  for i in $(seq 7); do
    mfccdir=exp/mfcc_perturbed/10$i
    data="data/10"$i
  	for x in train; do
      utils/perturb_data_dir_speed.sh 0.9 $data/${x} $data/temp1
      utils/perturb_data_dir_speed.sh 1.1 $data/${x} $data/temp2
      utils/copy_data_dir.sh --spk-prefix sp1.0- --utt-prefix sp1.0- $data/${x} $data/temp0
      utils/combine_data.sh $data/${x}_sp $data/temp0 $data/temp1 $data/temp2
      rm -r $data/temp0 $data/temp1 $data/temp2
      utils/validate_data_dir.sh --no-feats --no-text $data/${x}_sp

  		steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_16k.conf --cmd "$train_cmd" \
  				$data/${x}_sp exp/make_mfcc/${x} $mfccdir
  		steps/compute_cmvn_stats.sh $data/${x}_sp exp/make_mfcc/$x $mfccdir
  		utils/fix_data_dir.sh $data/${x}_sp
  	done
  done
fi


if [ $align -eq 1 ]; then
  for i in $(seq 7); do
    data="data/10"$i
    ali_dir="exp/10"$i"/ali"
    steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
                  $data/train_sp data/lang $modelDirectories/tri4 ${ali_dir}
  done
fi


if [ $mfccHiresForBaseData -eq 1 ]; then
  for i in $(seq 7); do
    mfccdir=exp/mfcc_hires/10$i
    data="data/10"$i
    for x in train dev; do
    	utils/copy_data_dir.sh $data/${x} $data/${x}_hires
    	steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_16k_hires.conf --cmd "$train_cmd" \
    			$data/${x}_hires exp/make_mfcc/${x}_hires $mfccdir
    	steps/compute_cmvn_stats.sh $data/${x}_hires exp/make_mfcc/${x}_hires $mfccdir
    	utils/fix_data_dir.sh $data/${x}_hires
    done
  done
fi


if [ $mfccHiresForSp -eq 1 ]; then
  for i in $(seq 7); do
    mfccdir=exp/mfcc_hires/10$i
    data="data/10"$i
    for x in train_sp; do
    	utils/copy_data_dir.sh $data/${x} $data/${x}_hires
    	utils/data/perturb_data_dir_volume.sh $data/${x}_hires
    	steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_16k_hires.conf --cmd "$train_cmd" \
    			$data/${x}_hires exp/make_mfcc/${x}_hires $mfccdir
    	steps/compute_cmvn_stats.sh $data/${x}_hires exp/make_mfcc/${x}_hires $mfccdir
    	utils/fix_data_dir.sh $data/${x}_hires
    done
  done


fi


if [ $ivector -eq 1 ]; then
  for i in $(seq 7); do
    data="data/10"$i
    for x in train_sp dev; do
      online_ivector_dir="exp/10"$i"/ivectors_"$x
    	steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    		$data/$x $modelDirectories/nnet3/extractor $online_ivector_dir
    done 
  done
fi



if [ $config -eq 1 ]; then

  mkdir -p $dir/configs
  num_targets=`tree-info exp/101/ali/tree 2>/dev/null | grep num-pdfs | awk '{print $2}'` || exit 1;
  

  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 input=Append(input@-2,input@-1,input,input@1,input@2, ReplaceIndex(ivector, t, 0)) dim=1024
  relu-renorm-layer name=tdnn2 dim=1024
  relu-renorm-layer name=tdnn3 input=Append(-1,2) dim=1024
  relu-renorm-layer name=tdnn4 input=Append(-3,3) dim=1024
  relu-renorm-layer name=tdnn5 input=Append(-3,3) dim=1024
  relu-renorm-layer name=tdnn6 input=Append(-7,2) dim=1024
  relu-renorm-layer name=tdnn_bn dim=$bnf_dim

  # adding the layers for diffrent language's output

  relu-renorm-layer name=prefinal-affine-0 input=tdnn_bn dim=1024
  output-layer name=output-0 dim=${num_targets} max-change=1.5

  relu-renorm-layer name=prefinal-affine-1 input=tdnn4 dim=1024
  output-layer name=output-1 dim=${num_targets} max-change=1.5

  relu-renorm-layer name=prefinal-affine-2 input=tdnn4 dim=1024
  output-layer name=output-2 dim=${num_targets} max-change=1.5

  relu-renorm-layer name=prefinal-affine-3 input=tdnn4 dim=1024
  output-layer name=output-3 dim=${num_targets} max-change=1.5


EOF

  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig \
    --config-dir $dir/configs/ \
    --nnet-edits="rename-node old-name=output-0 new-name=output"

fi


if [ $egs -eq 1 ]; then
  cmd=run.pl
  left_context=16
  right_context=12

  context_opts="--left-context=$left_context --right-context=$right_context"


  for i in $(seq 4); do
    data="data/10"$i

    ali_dir="exp/10"$i"/ali"
    online_ivector_dir="exp/10"$i"/ivectors_train_sp"

    transform_dir=${ali_dir}
    cmvn_opts="--norm-means=false --norm-vars=false"
    extra_opts=()
    extra_opts+=(--cmvn-opts "$cmvn_opts")
    extra_opts+=(--online-ivector-dir ${online_ivector_dir})
    extra_opts+=(--transform-dir $transform_dir)
    extra_opts+=(--left-context $left_context)
    extra_opts+=(--right-context $right_context)
    echo "$0: calling get_egs.sh for generating examples with alignments as output"


    steps/nnet3/get_egs.sh $egs_opts "${extra_opts[@]}" \
      --num-utts-subset 300 \
      --nj $nj \
        --samples-per-iter 100000 \
        --frames-per-eg 8 \
        --cmd "$cmd" \
        --generate-egs-scp true \
        --frames-per-eg 8 \
        $data/train_sp_hires ${ali_dir} $dir/egs_${i} || exit 1;
  done

fi


if [ $combineEgs -eq 1 ]; then
  if [ ! -z "$lang2weight" ]; then
      egs_opts="--lang2weight '$lang2weight'"
  fi
  common_egs_dir="$dir/egs_1 $dir/egs_2 $dir/egs_3 $dir/egs_4 $dir/egs"
  steps/nnet3/multilingual/combine_egs.sh $egs_opts \
    --cmd "$decode_cmd" \
    --samples-per-iter 400000 \
    $num_langs ${common_egs_dir[@]} || exit 1;
fi


if [ $train -eq 1 ]; then
  steps/nnet3/train_raw_dnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs 2 \
    --trainer.optimization.num-jobs-initial=$num_jobs_initial \
    --trainer.optimization.num-jobs-final=$num_jobs_final \
    --trainer.optimization.initial-effective-lrate=0.0015 \
    --trainer.optimization.final-effective-lrate=0.00015 \
    --trainer.optimization.minibatch-size=256,128 \
    --trainer.samples-per-iter=400000 \
    --trainer.max-param-change=2.0 \
    --trainer.srand=$srand \
    --feat-dir data/101/train_sp_hires \
    --feat.online-ivector-dir exp/101/ivectors_train_sp \
    --egs.dir $dir/egs \
    --use-dense-targets false \
    --targets-scp exp/101/ali \
    --cleanup.remove-egs false \
    --cleanup.preserve-model-interval 50 \
    --use-gpu true \
    --dir=$dir  || exit 1;
fi


if [ $priors -eq 1 ]; then
  for i in $(seq 4); do
    p=$((i-1))
    lang_dir=$dir/10$i
    mkdir -p  $lang_dir
    nnet3-copy --edits="rename-node old-name=output-$p new-name=output" \
    $dir/final.raw - | \
    nnet3-am-init exp/10$i/ali/final.mdl - \
    $lang_dir/final.mdl || exit 1;

    cp $dir/cmvn_opts $lang_dir/cmvn_opts || exit 1;
    echo "$0: compute average posterior and readjust priors for language $101-recognition."
    steps/nnet3/adjust_priors.sh --cmd "$decode_cmd" \
    --use-gpu true \
    --iter final --use-raw-nnet false --use-gpu true \
    $lang_dir $dir/egs_$i || exit 1;

  done

fi

if [ $decode -eq 1 ]; then
	
	dnn_beam=15.0
	dnn_lat_beam=8.0
	decode_stage=-2

  for i in $(seq 4); do
    data="data/10"$i
    score_opts="--skip-scoring false"

    ivectorsForDecode=0

    decode=$dir/10$i/decode_dev
    ivec_dir=exp/10$i/ivectors_dev

    if [ $ivectorsForDecode -eq 1 ]; then			
        steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
        	data/101-recognition/$x $modelDirectories/nnet3/extractor ${ivec_dir} || exit 1;
    fi
    ivector_opts="--online-ivector-dir ${ivec_dir}"

    mkdir -p $decode
    steps/nnet3/decode.sh --nj $nj --cmd "$decode_cmd" \
        --stage $decode_stage \
        --beam $dnn_beam --lattice-beam $dnn_lat_beam \
        $score_opts $ivector_opts \
        $modelDirectories/tri4/graph_sw1_tg $data/dev_hires $decode | tee $decode/decode.log
  done
fi


if [ $wer -eq 1 ]; then
  #for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 
  for x in exp/nnet3_*/*/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
  #for x in exp/*/*/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
fi