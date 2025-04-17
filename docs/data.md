# Why TouchDataset?

## **Background**  
- **Legacy Format Issues**:  
  - **Raw Format** ([[WeNet.raw]](), [[HuggingFace.]]()): Inefficient HDFS reads for small files. Caused memory overflow due to oversized `data.json` at hundreds of thousands of hours;
  - **Shard Format**: Solved above issues by packing data into tar files but failed to scale for **10 million-hour-level data** and **long-audio processing** needs.  
- **New Challenges**:  
  - **Storage Waste**: Redundant packaging for diverse training needs (e.g., short/long audio from the same source like hour-long podcasts).  
  - **Dynamic Slicing**: Requires on-the-fly audio segment extraction during training to improve data utilization.  

---

## **Core Requirements**
1. **Storage Optimization**:
   - Store **one copy of raw audio** to avoid redundancy.  
2. **Random Access**:  
   - Support dynamic slicing for both **unsupervised pre-training** (e.g., `audio[1:30]`) and **supervised training** (e.g., `audio[15:3000]`).  
3. **Decoupled Annotation**:  
   - Enable text/annotation updates (e.g., improved ASR or new event tags) without reprocessing audio.  
4. **Random Access**:  
   - Fast data retrieval via index files.  

---

## **New Data Format Design**  
- **File Structure**:  
  - **`audio.bin`**: Concatenated binary of all raw audio (color blocks represent distinct audio files).  
  - **`audio_index.bin`**: Byte offsets for each audio (e.g., `(0,3)` for red audio spanning bytes 0-3).  
  - **`info.bin`**: JSON annotations (source URLs, multi-version texts, segmentation metadata).  
  - **`info_index.bin`**: Byte indices for annotation data.  
  - **Other Modalities**: `video.bin` and `video_index.bin` follow the same structure.  
- **Advantages**:  
  - **Storage Decoupling**: Independent storage for audio, text, and video reduces redundancy.  
  - **Flexible Access**: Random reads and dynamic slicing via indices.  

---

## **Annotation Format**  
- **Key Fields**:  
  - `key`: Unique audio identifier (e.g., absolute path).  
  - `txt`: Default text (concatenation of all segment texts).  
  - `info`: Includes source URL, audio properties, and segment details (start/end times, Whisper/Paraformer transcripts).  
- **Dynamic Merging**: Adjacent segments can be merged based on start/end times to simulate varied training durations.  

---

## **Potential Issues & Solutions**  
1. **Long-Audio I/O Waste**:  
   - **Issue**: Low effective-slice ratios in compressed formats (e.g., m4a) require full decompression.  
   - **Solution**: Convert to WAV format (supports random access) to read only necessary parts.  
2. **Massive File Storage**:  
   - **Solution**: Shard storage (e.g., split `audio.bin` into multiple files).  

---

## **Implementation Progress**  
- **Performance Tests**:  
  - **TouchDataset**: Supports random reads, **10%-20% faster** than `ShardDataset`.  
  - **Advantage**: Outperforms legacy formats in speed and functionality.  

---

## **Conclusion**  
The new data format addresses **10 million-hour scalability** and **long-audio flexibility** through indexed storage (`audio.bin` + `*_index.bin`). Decoupled annotations, WAV-based random access, and sharding resolve redundancy and dynamic slicing challenges. Performance benchmarks confirm superiority over legacy solutions.
