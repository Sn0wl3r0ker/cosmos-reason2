### Server&test command
```bash
vllm serve nvidia/Cosmos-Reason2-2B   --allowed-local-media-path / --max-model-len 8192   --media-io-kwargs '{"video": {"num_frames": -1}}'   --reasoning-parser qwen3   --port 8888   --gpu-memory-utilization 0.7

cosmos-reason2-inference online --port 8888 -i prompts/cust.yaml --videos reason2_test_cut3.mp4
```
### video sampling
```bash
python -m cosmos_reason2_utils.script.video_overlay sample \
  --input reason2_test_cut3.mp4 \
  --infer-fps 2 \
  --sampled-dir cosmos_reason2_utils/cosmos_reason2_utils/video_overlay/sampled_frames \
  --metadata-out cosmos_reason2_utils/cosmos_reason2_utils/video_overlay/results/sampled_metadata.json
```

### video inference
```bash
python -m cosmos_reason2_utils.script.video_overlay infer   --metadata cosmos_reason2_utils/cosmos_reason2_utils/video_overlay/results/sampled_metadata.json   --backend online   --host localhost --port 8888   --results-out cosmos_reason2_utils/cosmos_reason2_utils/video_overlay/results/inference_results.json   --debug-dir cosmos_reason2_utils/cosmos_reason2_utils/video_overlay/debug_frames   --prompt "You are an Autonomous Mobile Robot with wheels, and have a 6 Dof Robot Arm that can reach or press. Determine the next movement step by step. Your objective is to go another floor."
```
### overlay
```bash
python -m cosmos_reason2_utils.script.video_overlay overlay \
  --input reason2_test_cut3.mp4 \
  --results cosmos_reason2_utils/cosmos_reason2_utils/video_overlay/results/inference_results.json \
  --segments-out cosmos_reason2_utils/cosmos_reason2_utils/video_overlay/results/overlay_segments.json \
  --output cosmos_reason2_utils/cosmos_reason2_utils/video_overlay/output/overlay.mp4
```