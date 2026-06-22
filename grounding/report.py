from models.research import LinkedAnswer

def render(linked: LinkedAnswer) -> str:
    rendered_sentences = []
    footnote_counter = 1
    footnotes = []
    
    for ls in linked.sentences:
        if ls.grounded and ls.evidence:
            span = ls.evidence[0]
            # Formulate the markdown footnote
            citation_marker = f"[^{footnote_counter}]"
            footnote_text = f"{citation_marker}: {span.source_file}, chunk {span.chunk_index}"
            
            rendered_sentences.append(f"{ls.sentence} {citation_marker}")
            footnotes.append(footnote_text)
            footnote_counter += 1
        else:
            rendered_sentences.append(f"[UNGROUNDED] {ls.sentence}")
            
    report_body = " ".join(rendered_sentences)
    
    if footnotes:
        report_body += "\n\n---\n**Sources**\n"
        report_body += "\n".join(footnotes)
        
    return report_body
