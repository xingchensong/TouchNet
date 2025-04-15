#!/bin/bash
# Copyright [2024-04-09] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>

cuda_version=12.6.3
driver_version=560.35.05
cudnn_version=9.5.1.17
prefix=/bucket/output/jfs-hdfs/user/xingchen.song/tools/cuda

echo "start download cuda ${cuda_version} & cudnn ${cudnn_version}"
wget https://developer.download.nvidia.com/compute/cuda/${cuda_version}/local_installers/cuda_${cuda_version}_${driver_version}_linux.run
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-${cudnn_version}_cuda12-archive.tar.xz
echo "end download cuda ${cuda_version} & cudnn ${cudnn_version}"

echo "start install cuda ${cuda_version}"
tmp_dir=${prefix}/tmp
rm -rf ${tmp_dir}
mkdir -p ${prefix}/cuda-${cuda_version}_cudnn-${cudnn_version}
mkdir -p ${tmp_dir}
./cuda_${cuda_version}_${driver_version}_linux.run \
  --silent \
  --toolkit \
  --installpath=${prefix}/cuda-${cuda_version}_cudnn-${cudnn_version} \
  --no-opengl-libs \
  --no-drm \
  --no-man-page \
  --tmpdir=${tmp_dir}
echo "end install cuda ${cuda_version}"

echo "start install cudnn ${cudnn_version}"
rm -rf ${tmp_dir}
mkdir -p ${tmp_dir}
tar xvf cudnn-linux-x86_64-${cudnn_version}_cuda12-archive.tar.xz --strip-components=1 -C ${tmp_dir}
cp ${tmp_dir}/include/cudnn* ${prefix}/cuda-${cuda_version}_cudnn-${cudnn_version}/include
cp ${tmp_dir}/lib/libcudnn* ${prefix}/cuda-${cuda_version}_cudnn-${cudnn_version}/lib64
chmod a+r ${prefix}/cuda-${cuda_version}_cudnn-${cudnn_version}/include/cudnn*.h ${prefix}/cuda-${cuda_version}_cudnn-${cudnn_version}/lib64/libcudnn*
echo "end install cudnn ${cudnn_version}"
