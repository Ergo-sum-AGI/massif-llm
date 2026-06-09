def _wrap_dataset(dataset, tokenizer, config: MASSIFConfig, max_steps: int):
    """Wrap a HuggingFace streaming dataset into batches of input_ids efficiently."""
    seq_len = config.max_seq_len
    batch_size = config.batch_size
    buffer = []
    
    # Initialize state tracking cleanly
    if not hasattr(_wrap_dataset, '_batch'):
        _wrap_dataset._batch = []
    if not hasattr(_wrap_dataset, '_step_count'):
        _wrap_dataset._step_count = 0

    for example in dataset:
        text = example.get('text', '')
        if not text:
            continue
        
        # FIXED: Added truncation and max_length to prevent the warning
        ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=seq_len)
        buffer.extend(ids)

        # Process chunks out of the buffer
        cursor = 0
        while len(buffer) - cursor >= seq_len + 1:
            chunk = buffer[cursor : cursor + seq_len + 1]
            cursor += seq_len + 1
            
            input_ids = torch.tensor(chunk[:seq_len], dtype=torch.long).unsqueeze(0)
            _wrap_dataset._batch.append(input_ids)

            if len(_wrap_dataset._batch) == batch_size:
                batch_tensor = torch.cat(_wrap_dataset._batch, dim=0)
                _wrap_dataset._batch = []
                yield {'input_ids': batch_tensor}
                
                _wrap_dataset._step_count += 1
                if _wrap_dataset._step_count >= max_steps:
                    return

        # Clear out the consumed portion of the buffer
        if cursor > 0:
            buffer = buffer[cursor:]
